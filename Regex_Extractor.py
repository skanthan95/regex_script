import pyspark.sql.functions as F
from pyspark.sql.functions import trim, col, udf, lit, desc, concat_ws, size, length, array, coalesce
from pyspark.sql.types import IntegerType, ArrayType, StructType, FloatType, StructField, StringType
import re
import collections
import os
from azureml.core import Workspace, Datastore, Dataset
from azureml.core.compute import ComputeTarget, AmlCompute, ComputeInstance
from azureml.data.datapath import DataPath
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
import itertools

# Defining all of the regex searches, case-insensitive

'''
Improvements:

- Replaced previous size regex with more compact version (was able to combine many of the groups)
     Improved size regex to capture existing size formats, plus cases like these which were previously missed:
     - 6cm (no space between 6 and cm)
     - 3 range
     - 1-2 mm (most frequently missed format so far)
     - sub-4mm
     The new size regex searches for 3D measurements first, then 2D, 1D, followed by the edge cases (if we search for 1-2D first, they'd be subsumed inside 3D and those wouldn't be extracted correctly)
     It doesn't match strings that are followed by axial, medial, or sagittal

     It improves accuracy on the other terms too, since if a size term isn't identified, the substring isn't evaluated further for other descriptors

- Corrected misspelling in descriptor regex ("party\s+solid") and expanded it to (partly\s*?\-?solid), added "calcified", "partly\s*?\-?calcified", removed duplicate "partly\s+solid".
  Expanded "sub solid" to sub\s*?\-?solid. Removed "ground glass" variations and replaced with ground\s*?\-?glass which captures them all

- Added clinician-recommended terms to progression. ** Added basic negation filter to descriptor and progression regexes; if they're preceded by no/not, they aren't counted as a match

- Improved follow-up time regex to capture more things (e.g., "1 year" vs "one year" vs "one-year" followup). (added previously-excluded hyphens, factored out common terms). Added
  more numbers to choose from since I noticed that some, like twelve months from the Society guidelines, were being missed
'''

flags = re.IGNORECASE

flags = re.IGNORECASE

search_size = re.compile(r"(?!.*(axial|medial|sagittal))((small)|(sub\-?centimeter)|(tiny)|(volume)|(diameter)|(\d+\.?\d*\s*(mm|cm)?\s*x\s*\d+\.?\d*\s*(mm|cm)?\s*x\s*\d+\.?\d*\s*(mm|cm))|(\d+\.?\d*\s*(mm|cm)?\s*x\s*\d+\.?\d*\s*(mm|cm))|(\d+\.?\d*\s*(mm|cm)|(([a-z]+)?\d*\s*-\s*\d*\s*(mm|cm))|(\d+\-?\s*?range)|(\d+\s+to\s+\d+ (mm|cm))))", flags)
search_term = re.compile(r'((pulmonary\s+nodules?)|(lung\s+nodules?)|(\bnodules?)|(\bmass\b)|(\bmasses\b)|(lesions?)|(opacity)|(opacities)|(\bGGO\b)|(neoplasms?))',flags)
search_location = re.compile(r'\w+\-?\s*\w+\-?\s*lobe', flags)
search_descriptor = re.compile(r'((calcified)|(calcifications?)|(sub\s*?\-?solid)|(ground\s*?\-?glass)|(partly\s*?\-?solid)|(spiculated)|(spiculation)|((?<!\-)solid\b))',flags)
search_progression = re.compile(r'((Enlarged)|(enlarging)|(increased?)|(\bnew\b)|(enlargement)|(progression)|(\bevolution\b)|(suspicious)|(bigger)|(doubling\s*?\-?\s*time)|(prominent)|(\bconspicuous\b)|(meta\-?static)|(worrisome))',flags)
search_recommend = re.compile(r'((recommended)|(Recommendation)|(Recommend))',flags)
search_recommend_task = re.compile(r'((ct\s+chest)|(chest\s+ct)|(CT\s?of\s?the\s?chest)|(PET/CT)|(PET\s+CT)|(\bPET\b)|(\bCTs)|(\bCT\b)|(\bMRI)|(\bMR\b)|(biopsy)|(consultation)|(pulmonary\s+consultation)|(tumor\s+board)|(\brepeat\b)|(nodule\s+board)|(bronchoscopy)|(direct\s+visualization)|(examination))',flags)
search_followup = re.compile(r'((FOLLOW-UP)|(followup)|(follow\s+up)|(F/U)|(f-u)|(\bfu\b))',flags)
search_followup_time = re.compile(r'((\d+\s*(or|and)\s*\d+\s*(weeks?|months?|years?))|(\d+\s*months\s*to\s*a\s*year)|(\d+\s*(to|\-)\s*\d+\s*(weeks?|months?|years?))|(\d+\s*\-?\s*(weeks?|months?|years?))|((o(?i)(LUNG(\-|\s*?)?RADS?\s*?(\[20\d{2}\])?(\(V\d\.?\d?\))?(:|\-)?\s*?(Assessment|(score\s*(of)?))?:?\s*?((\d+\s*(\w+)?)|((CATEGORY|RECOMMENDATION):?\s*?(category)?\s*?\d+(\w+)?)|(MODIFIER):?\s+?\S+(\s*\w+)?))|(LR\s*\d\w?:\s*(PROBABLY|NEGATIVE|INCOMPLETE|SIGNIFICANT|SUSPICIOUS(\s*WITH\s*ADDITIONAL\s*FEATURES)?|VERY))ne|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\-?\s*(weeks?|months?|years?))|(\d+\s*(to|\-)\s*\d+\s*(weeks?|months?|years?))|(annual))',flags)
search_lungrad = re.compile(r"(?i)(LUNG(\-|\s*?)?RADS?\s*?(\[20\d{2}\])?(\(V\d\.?\d?\))?(:|\-)?\s*?(Assessment|(score\s*(of)?))?:?\s*?((\d+\s*(\w+)?)|((CATEGORY|RECOMMENDATION):?\s*?(category)?\s*?\d+(\w+)?)|(MODIFIER):?\s+?\S+(\s*\w+)?))|(LR\s*\d\w?:\s*(PROBABLY|NEGATIVE|INCOMPLETE|SIGNIFICANT|SUSPICIOUS(\s*WITH\s*ADDITIONAL\s*FEATURES)?|VERY))", flags)


def is_valid(phrase, match):
    '''
    Helper function to find_index_match
    (Applies to all category searches except lungrads)

    Input:
        Phrase: str
        match: re match object

    Takes a phrase from the find_index_match loop and searches
    for the word in that phrase right before the match.
    If that (cleaned) word is no/not/without/denies, treat
    the match as negated and discount it

    Returns bool (True if no negation term preceding match OR match is the first word in the phrase,
    in which it assumes no negation preceding; False if negation found)

    Recall: This is meant to be an extremely simple negation check. it will only work if the word
    directly preceding the match is "no/not/without/denies". instances like 
    "there are no suspicious calcifications" will ONLY negate "suspicious" but still flag 
    "calcifications". Will build on negation method in future release
    '''
    try:
        pre_word = phrase.partition(match.group())[0].split()[-1].strip()
        pre_word = re.sub(r"\\r|\\n|\\v|\\t|[;':#$%&@*^\\-_=+.,?!]", "", pre_word) # removing puncutation and weird spacing from previous word for clear evaluation
        if re.search(r"(?i)(\bnot\b|\bno\b|\bwithout\b|\bdenies\b)", pre_word):
            return False
        return True
    except:
        return True
 

def find_lungrad_match_index(note, regex_list):
    '''
    Takes a note and list of regex compile objects, like the find_match_index function
    But for now, this will only ever be for the single lungrad regex.
    This function scans the entire note for lungrad matches, does NOT evaluate phrase by phrase
    (we found that to be less accurate, since we split on \.\s and that caused some false negatives)

    Returns matches and indices in the same format as find_match_index, below
    '''
    
    #Designed for note-level matches (find_match_index is designed for phrase-level matches)

    matches = [[] for i in range(len(regex_list))] # [[],[]]
    indices = [[] for i in range(len(regex_list))] 

    if note is not None:
        for i,reg in enumerate(regex_list):
    # Save matches per category into initialized designated sublist in store_matches_strings
            matches[i].extend([match.group() for match in re.finditer(reg, note) if 'N/A' not in match.group() and not re.search(r'(?i)Not\s*applicable',  match.group()) and not 'None' in  match.group() and not re.search(r'(?<!\[)(\r?\s?)20\d{2}', match.group())])
            indices[i].extend([[match.start(0), match.start(0) + len(match.group())] for match in re.finditer(reg, note) if 'N/A' not in  match.group() and not re.search(r'(?i)Not\s*applicable',  match.group()) and not 'None' in match.group() and not re.search(r'(?<!\[)(\r?\s?)20\d{2}', match.group())])
    else:
        note = ""

    return matches, indices




def standardize_lungrads(list_of_extracted_lungrads):
    '''
    Once the lungrads are extracted from a note, they are passed here
    to be standardized. This is a helper function called by extract_worst_lungrad.
    '''
    # Removing lungrads that say "N/A" or "None" or include year as description (noise)
    modified_input = [rad for rad in list_of_extracted_lungrads if 'N/A' not in rad and not re.search(r'(?i)Not\s*applicable', rad) and not 'None' in rad and not re.search(r'(?<!\[)(\r?\s?)20\d{2}', rad)]
    #print("modified input:", modified_input)

    standardized_lungrads = []

    for i, lungrad in enumerate(modified_input):

        # Removing punctuation and weird spacing
        lungrad = re.sub(r"\\r|\\n|\\v|\\t|[;':#$%&@*^\\-_=+.,?!]", "", lungrad)
        #print("cleaned lungrad:", lungrad)

        # Standardization criteria for each lungrad type
        if re.search(r'\b0\b', lungrad) or re.search(r'(?i)\bincomplete\b', lungrad):
            standardized_lungrads.append('LUNGRADS 0: INCOMPLETE')
        elif re.search(r'\b1\b', lungrad) or re.search(r'(?i)\bnegative\b', lungrad):
            standardized_lungrads.append('LUNGRADS 1: NEGATIVE')
        elif re.search(r'\b2\b', lungrad) or re.search(r'(?i)(?<!probably\s)\bBenign\b', lungrad): # regex explanation: benign if not preceded by probably
            standardized_lungrads.append('LUNGRADS 2: BENIGN APPEARANCE OR BEHAVIOR')
        elif re.search(r'\b3\b', lungrad) or re.search(r'(?i)\bprobably\s*Benign\b', lungrad):
            standardized_lungrads.append('LUNGRADS 3: PROBABLY BENIGN')
        elif re.search(r'\bS\b', lungrad) or re.search(r'(?i)\bSIGNIFICANT INCIDENTAL FINDING\b', lungrad):
            standardized_lungrads.append('LUNGRADS S: SIGNIFICANT INCIDENTAL FINDING')
        elif re.search(r'\b4A\b', lungrad) or re.search(r'(?i)(?<!very\s)\bsuspicious\b(?!\W+with additional features\b)', lungrad): # regex explanation: suspicious if not preceded by 'very' and not followed by 'with additional features'
             standardized_lungrads.append('LUNGRADS 4A: SUSPICIOUS')
        elif re.search(r'\b4B\b', lungrad) or re.search(r'(?i)\bvery\s*suspicious\b', lungrad):
            standardized_lungrads.append('LUNGRADS 4B: SUSPICIOUS')
        elif re.search(r'\b4X\b', lungrad) or re.search(r'(?i)SUSPICIOUS WITH ADDITIONAL FEATURES', lungrad):
            standardized_lungrads.append('LUNGRADS 4X: SUSPICIOUS WITH ADDITIONAL FEATURES')
        else:
            standardized_lungrads.append("OTHER")


    # The length of the lungrads list must be the same as the length of the standardized lungrad list
    assert len(standardized_lungrads) == len(modified_input) 


    #print(standardized_lungrads)
    return standardized_lungrads


def extract_most_serious_lungrad(list_of_lungrad_matches):
    '''
    Takes a list of extracted lungrads for a given note.
    Calls the helper standardization function.
    Then, selects the 'worst' lungrad.
    '''

    standardized_list_of_lungrad_matches = standardize_lungrads(list_of_lungrad_matches)

    # Define lungrad values and their severity order
    lungrad_order = {'LUNGRADS 4X: SUSPICIOUS WITH ADDITIONAL FEATURES':7,'LUNGRADS 4B: SUSPICIOUS': 6, 'LUNGRADS 4A: SUSPICIOUS': 5,
     'LUNGRADS 3: PROBABLY BENIGN': 4, 'LUNGRADS 2: BENIGN APPEARANCE OR BEHAVIOR': 3, 'LUNGRADS 1: NEGATIVE' : 2,
      'LUNGRADS 0: INCOMPLETE':1, 'LUNGRADS S: SIGNIFICANT INCIDENTAL FINDING': 0, 'OTHER':-1}

    # Determine the most serious lungrad value based on severity order
    if len(standardized_list_of_lungrad_matches) > 0:
        most_serious_lungrad = max(standardized_list_of_lungrad_matches, key=lambda x: lungrad_order[x])
    else:
        most_serious_lungrad = None

    return most_serious_lungrad


extract_worst_lungrad_udf = F.udf(extract_most_serious_lungrad, StringType())



def compare_lung_columns(diagnostic_code_list, worst_lungrad_value):
    '''
    diagnostic_code_list: list of diagnostic codes (list of strings). In practice,
    this will be the dataframe's diagCode list column value for given note/row

    worst_lungrad_value: worst_lungrad string. In practice, this will be the 
    dataframe's worst_lungrad string value for given note/row
    '''

    # Deal with None values in worst_lungrad column and diagnostic codes column
    # Standardizing diagnostic codes and worst lungrad by making them lowercase

    if worst_lungrad_value is None:
        worst_lungrad_value = 'null'

    remove_nulls = ['null' if x is None else x for x in diagnostic_code_list]
    lowercase_list = [val.lower() for val in remove_nulls]

    # If the worst lungrad value is null but there is a lungrad diagnostic code
    if worst_lungrad_value == 'null':
        if len([val for val in lowercase_list if 'lungrad' in val]) > 0:
            return 1
        # Case where worst lungrad is null and no lungrad diagnostic code is irrelevant
        
    # If the worst lungrad value is not null
    else:
        lowercase_worst_lungrad = worst_lungrad_value.lower()
        # If the worst lungrad matches any of the values in the diagnostic code list
        if any(lowercase_worst_lungrad in s for s in lowercase_list):
            return 2
        # If the worst lungrad value does not match any of the values in the diagnostic code list, BUT there is a different lungrad diagnostic code
        elif not any(lowercase_worst_lungrad in s for s in lowercase_list) and len([val for val in lowercase_list if 'lungrad' in val]) > 0:
            return 3
        # If worst lungrad exists but the diagnostic code(s) do not relate to lungrads (covers missing diagnostic codes too)
        elif not any(lowercase_worst_lungrad in s for s in lowercase_list) and len([val for val in lowercase_list if 'lungrad' in val]) == 0:
            return 4
        else:
            5


compare_lungrads_udf = F.udf(compare_lung_columns, StringType())


def convert_diag_codes(arr):
    '''
    There are lungrad diagnostic codes that are very similar; they need to be standardized
    (This looks at all the lungrads in the diagCode list column and standardizes them)
    '''

    diag_replace = {'LUNGRADS 2: BENIGN NODULE APPEARANCE OR BEHAVIOR':'LUNGRADS 2: BENIGN APPEARANCE OR BEHAVIOR',
'LUNGRADS 3: PROBABLY BENIGN NODULE':'LUNGRADS 3: PROBABLY BENIGN',
'LUNGRADS 4A: SUSPICIOUS NODULE':'LUNGRADS 4A: SUSPICIOUS',
'LUNGRADS 4B: SUSPICIOUS NODULE':'LUNGRADS 4B: SUSPICIOUS',
'LUNGRADS 4X: SUSPICIOUS NODULE WITH ADDITIONAL FEATURES':'LUNGRADS 4X: SUSPICIOUS WITH ADDITIONAL FEATURES'}

    return [diag_replace.get(item, item) for item in arr]


convert_diag_udf = udf(convert_diag_codes, ArrayType(StringType()))


def find_match_index(note, regex_list):

    '''
    Input: a note
           regex_list: a list of re.compile objects to use for regex match extraction (e.g., [search_size, search_term])
    Returns:
           Two lists; the first contains matches from each regex category. the second contains the indices corresponding to the matches from each category.

    Sample function call:
           find_match_index("There is a 6mm nodule. There is also a 5.6mm x 8.9mm mass.", [search_size, search_term])

    Output: 
           ([['6mm', '5.6mm x 8.9mm'], ['nodule', 'mass']], [[[11, 14], [39, 52]], [[15, 21], [53, 57]]])

           Notice how all matches are captured in one list, but each category goes into its own sublist (same for indices)
    '''

    # Split note into phrases/sentences 

    # If note is None/null, return store_matches_strings/store_matches_indices as-is (empty)
    if note is None:
        split_note = ""
    # Only proceed with splitting note and moving on to regex matching if note length > 0
    elif len(note) > 0:
        split_note = re.split(r'\.\s+', note)
    # Handling other forms of missing notes
    else:
        split_note = ""



    # Each category's results will go in its own list inside these initialized lists, for easier separation into distinct columns
    store_matches_strings = [[] for i in range(len(regex_list))] 
    store_matches_indices = [[] for i in range(len(regex_list))]

    # Loop over each sentence/phrase
    for phrase in split_note: 

        # Only extract regex matches if all regex categories flag a match in the phrase (e.g., don't evaluate phrase if size in phrase but term is not)
        if all(reg.search(phrase) for reg in regex_list): 
            # Set the current phrase start index relative to original note
            phrase_start = len(note.split(phrase, 1)[0])


            # Extract regex matches and indices for each category in regex_list
            for i,reg in enumerate(regex_list):
                # Save matches per category into initialized designated sublist in store_matches_strings
                store_matches_strings[i].extend([match.group() for match in re.finditer(reg, phrase) if is_valid(phrase, match)])

                # Save indices of each match per category into initialized designated sublist in store_matches_indices
                # Start = phrase_start + match start relative to phrase, End = phrase_start + length of match itself. This gives us start/end indices (relative to note)
                store_matches_indices[i].extend([[phrase_start + match.start(0), phrase_start + match.start(0) + len(match.group())] for match in re.finditer(reg, phrase) if is_valid(phrase, match)])

                        
    return store_matches_strings, store_matches_indices



# Output schema for storing the results each time run find_match_index
# (recall it's two nested lists)



output_schema = StructType([
    StructField("string_matches", ArrayType(ArrayType(StringType()))),
    StructField("match_indices", ArrayType(ArrayType(ArrayType(IntegerType()))))])



def find_lungrad_info_udf(regex_list, schema):
    '''
    UDF for find_match_index function. Takes regex_list 
    and a schema to store the output (output_schema)
    '''
    return udf(lambda l: find_lungrad_match_index(l, regex_list), schema)



def find_match_index_udf(regex_list, schema):

    '''
    UDF for find_match_index function. Takes regex_list 
    and a schema to store the output (output_schema)
    '''
    return udf(lambda l: find_match_index(l, regex_list), schema)



def extract_vals(label_list, df, note_type_list, col_list_1, col_list_2):   

    '''
    A helper function that takes the regex matches/indices from each category
    and puts them in their own aptly-named columns. Each category will get two 
    columns: matches vs indices for each match.

    Input:
        label_list: the regex categories as strings (e.g., ['size', 'term']). Used to create column names
        df: spark dataframe (containing imp_string_matches/indices and rep_string_matches/indices cols from create_flags())
        note_type_list: ['imp', 'rep'] for 'impression' vs. 'report' text. Used to create column names ** must have length of 2
        col_list_1: ['imp_string_matches', 'imp_index_matches'] (impression match col names) ** must have length of 2
        col_list_2: ['rep_string_matches', 'rep_index_matches'] (report match col names) ** must have length of 2

    Returns:
        df: input spark dataframe but with the individual regex category columns added.

    Assumptions: col_list_1/2 and note_type_list will only ever contain two values (since we're only working with two types of notes)
    '''

    for i, val in enumerate(label_list):

        category_name = val

        # This loop will create 4 new columns per regex category (match and index info for impression, match and index info for report).
        # If we're evaluating label_list ['size', 'term'], the new columns will be:
        # 'imp_size_matches', 'imp_size_indices, 'imp_term_matches', 'imp_term_indices', 'rep_size_matches', 'rep_size_indices', 'rep_term_matches', 'rep_term_indices'

        df = df.withColumn(note_type_list[0] + "_" + category_name + "_matches", col(col_list_1[0]).getItem(i))
        df = df.withColumn(note_type_list[0] + "_" + category_name + "_indices", col(col_list_1[1]).getItem(i))
        df = df.withColumn(note_type_list[1] + "_" + category_name + "_matches", col(col_list_2[0]).getItem(i))
        df = df.withColumn(note_type_list[1] + "_" + category_name + "_indices", col(col_list_2[1]).getItem(i))

    return df


def create_flags(df, regex_list, label_list, schema, flag_vals, lungrad=False):

    '''
    Creates the match/index columns for the input regex categor(ies) and calls extract_vals()
    to put each match category/index list into its own column. Then, creates a flag to indicate
    whether there are matches for the regex categor(ies) in just impression, report,
    or both.

    Input:
          df: spark dataframe (original notes df)
          regex_list: same param as in find_match_index()
          label_list: same param as in extract_vals()
          schema: output_schema
          flag_vals: list of custom flag values. **  must be in this order and must contain 3 values **: impression only, report only, both. e.g., ['IS', 'RS', 'IS_RS']

    Returns:
         df with rows where there was a match for the input regex categor(ies), with each match category put into a separate column
         and a flag column indicating which note type contained a match
         
    '''

    # Get regex match/index output for impression text
    if not lungrad:
        df_imp = df.withColumn('imp_output', find_match_index_udf(regex_list, schema)('ImpressionText'))
    else:
        df_imp = df.withColumn('imp_output', find_lungrad_info_udf(regex_list, schema)('ImpressionText'))

    # Expand the impression match/index output into two separate columns ("string_matches" and "index_matches", per the output_schema;
    # adding 'imp' prefix to them)
    explode_imp = df_imp.select('RadiologyRegisteredExamSID', 'RadiologyNuclearMedicineReportSID', 'diagCode_code_lst',
       'ImpressionText', 'ReportText', 'imp_output.*')\
       .withColumnRenamed("string_matches", "imp_string_matches")\
       .withColumnRenamed("match_indices", "imp_index_matches")


    # Get regex match/index output for report text
    if not lungrad:
        df_rep = explode_imp.withColumn('rep_output', find_match_index_udf(regex_list, schema)('ReportText'))
    else:
        df_rep = explode_imp.withColumn('rep_output', find_lungrad_info_udf(regex_list, schema)('ReportText'))

    # Expand the report match/index output into two separate columns ("string_matches" and "index_matches", per the output_schema;
    # adding 'rep' prefix to them)
    explode_rep = df_rep.select('RadiologyRegisteredExamSID', 'RadiologyNuclearMedicineReportSID', 'diagCode_code_lst',
       'ImpressionText', 'ReportText', 'imp_string_matches', 'imp_index_matches', 'rep_output.*')\
       .withColumnRenamed("string_matches", "rep_string_matches")\
       .withColumnRenamed("match_indices", "rep_index_matches")
  
    # Call extract_vals() function to put each regex category's matches/indices into separate columns
    extract = extract_vals(label_list, explode_rep, ['imp', 'rep'], ['imp_string_matches', 'imp_index_matches'], ['rep_string_matches', 'rep_index_matches'])

    if 'imp_rad_matches' in extract.columns:
        #extract = extract.withColumn('worst_lungrad', extract_worst_lungrad_udf(F.col('imp_rad_matches')))
        extract = extract.withColumn('worst_lungrad', F.when(col("imp_rad_matches")==F.array(), 
        extract_worst_lungrad_udf(F.col('rep_rad_matches'))).otherwise(extract_worst_lungrad_udf(F.col("imp_rad_matches"))))

        extract = extract.withColumn('lungrad_compare_flag', compare_lungrads_udf(F.col('diagCode_code_lst'),F.col('worst_lungrad')))
        extract = extract.na.fill({'lungrad_compare_flag': 0})
        extract = extract.withColumn("lungrad_compare_flag", extract["lungrad_compare_flag"].cast(IntegerType()))

    
    # Creating a flag column for the regex category based on the label_list input (e.g., size_term_flag)
    flag_label = "_".join([str(item) for item in label_list]) + "_flag"

    # Assigning value to flag column depending on whether there's a match in impression and/or report text per row
    imp_rep_flag = extract.withColumn(flag_label, \
    F.when((size(col("imp_string_matches")[0]) > 0) & (size(col("rep_string_matches")[0]) == 0), flag_vals[0])\
    .when((size(col("rep_string_matches")[0]) > 0) & (size(col("imp_string_matches")[0]) == 0), flag_vals[1])\
    .when((size(col("imp_string_matches")[0]) > 0) & (size(col("rep_string_matches")[0]) > 0), flag_vals[2]))


    # Drop rows that have no val in Flag col (so no match output for either impression or report; this follows the logic that Stringer used in his version)
    #imp_rep_flag_final = imp_rep_flag.filter(length(flag_label) > 0)
  
    # Return new dataframe without the initial imp/rep match/index cols, now that we have the expanded ones set
    return imp_rep_flag.drop('imp_string_matches',
       'imp_index_matches', 'rep_string_matches', 'rep_index_matches', 'Impression') 


def run_step_2(df):
    '''
    Main step 2 function. Takes output df from step 1 and checks for matches in all regex categories,
    one group at a time (first size & term, then rec & rec task, follow-up/follow-up time, descriptor, progression, and location).

    Joins the df outputs from each regex group search, drops rows that don't have any matches across all categories.
    Returns df
    '''

    # Filtering step 1 df to only rows where Impression OR Report have the following lung-related terms/prefixes 
    df = df.filter(df['ImpressionText'].rlike(('(?i)(lung|pulmonary|pneumon|bronch|alveo|pleura|respir)'))|df['ReportText'].rlike(('(?i)(lung|pulmonary|pneumon|bronch|alveo|pleura|respir)')))
    
    # Standardizing the diagCode column 
    df = df.withColumn("diagCode_code_lst", convert_diag_udf(df["diagCode_code_lst"]))

    # Extracting regex matches/indices and creating flags for each regex category/combination of interest
    # Dropping ImpressionText and ReportText at this stage for each category prevents join problem later on
    size_term = create_flags(df, [search_size, search_term], ['size', 'term'], output_schema, ['IS', 'RS', 'IS_RS']).drop('ImpressionText', 'ReportText', 'diagCode_code_lst')
    rec = create_flags(df, [search_recommend], ['rec'], output_schema, ['IR', 'RR', 'IR_RR']) # ** Dropping Impression/Report later, it's needed for recommend_task check first
    # Extracting 'rec task' matches only from notes that already had 'rec' matches
    rec_task = create_flags(rec, [search_recommend_task], ['rec_task'], output_schema, ['IRT', 'RRT', 'IRT_RRT']).drop("ImpressionText", "ReportText", 'diagCode_code_lst') # Searching for rec task within subset of rec matches
    followup = create_flags(df, [search_followup], ['fu'], output_schema, ['IF', 'RF', 'IF_RF']) # ** Dropping Impression/Report later, it's needed for follow # Searching for followup time within subset of rec matchesup_time check first
    # Extracting 'followup time' matches only from notes that already had 'followup' matches (besides this and rec_task, all other categories searched in original step 1 df)
    followup_time = create_flags(followup, [search_followup_time], ['fu_time'], output_schema, ['IFT', 'RFT', 'IFT_RFT']).drop('ImpressionText', 'ReportText', 'diagCode_code_lst')
    descr = create_flags(df, [search_descriptor], ['desc'], output_schema, ['ID', 'RD', 'ID_RD']).drop('ImpressionText', 'ReportText', 'diagCode_code_lst') # broader search than Stringer's 
    prog = create_flags(df, [search_progression], ['prog'], output_schema, ['IP', 'RP', 'IP_RP']).drop('ImpressionText', 'ReportText', 'diagCode_code_lst')
    loc = create_flags(df, [search_location], ['loc'], output_schema, ['IL', 'RL', 'IL_RL']).drop('ImpressionText', 'ReportText', 'diagCode_code_lst')
    lungrad = create_flags(df, [search_lungrad], ['rad'], output_schema, ['ILR', 'RLR', 'ILR_RLR'], lungrad=True).drop('ImpressionText', 'ReportText', 'diagCode_code_lst')


    # Joining the dataframes together with respect to the original input dataframe (** Need to check for unexpected behaviors
    # but so far, seems consistent; unionAll would be for cases where all the columns are the same but they aren't here
    # and I thought that was for row concatenation, not joining?)

    df_size = df.join(size_term, ["RadiologyRegisteredExamSID", "RadiologyNuclearMedicineReportSID"], how = 'left')
    df_size_rec = df_size.join(rec.drop('ImpressionText', 'ReportText', 'diagCode_code_lst'), ["RadiologyRegisteredExamSID", "RadiologyNuclearMedicineReportSID"], how = 'left')
    df_size_rec_rectask = df_size_rec.join(rec_task, ["RadiologyRegisteredExamSID", "RadiologyNuclearMedicineReportSID"], how = 'left')
    df_size_rec_rectask_fu = df_size_rec_rectask.join(followup.drop('ImpressionText', 'ReportText', 'diagCode_code_lst'), ["RadiologyRegisteredExamSID", "RadiologyNuclearMedicineReportSID"], how = 'left')
    df_size_rec_rectask_fu_fut = df_size_rec_rectask_fu.join(followup_time, ["RadiologyRegisteredExamSID", "RadiologyNuclearMedicineReportSID"], how = 'left')
    df_size_rec_rectask_fu_fut_desc = df_size_rec_rectask_fu_fut.join(descr, ["RadiologyRegisteredExamSID", "RadiologyNuclearMedicineReportSID"], how = 'left')
    df_size_rec_rectask_fu_fut_desc_prog = df_size_rec_rectask_fu_fut_desc.join(prog, ["RadiologyRegisteredExamSID", "RadiologyNuclearMedicineReportSID"], how = 'left')
    df_size_rec_rectask_fu_fut_desc_prog_loc = df_size_rec_rectask_fu_fut_desc_prog.join(loc, ["RadiologyRegisteredExamSID", "RadiologyNuclearMedicineReportSID"], how = 'left')
    df_size_rec_rectask_fu_fut_desc_prog_loc_lungrad = df_size_rec_rectask_fu_fut_desc_prog_loc.join(lungrad, ["RadiologyRegisteredExamSID", "RadiologyNuclearMedicineReportSID"], how = 'left')
    

    # Creating concatenated flag call with all permutations of flags across all categories

    flag_cols = [x for x in df_size_rec_rectask_fu_fut_desc_prog_loc_lungrad.columns if 'flag' in x]

    all_matches = df_size_rec_rectask_fu_fut_desc_prog_loc_lungrad.withColumn("Final_Flag", concat_ws('_',*flag_cols)) 
   
    # Dropping rows where the concatenated flag is null (i.e., no matches across all categories)
    drop_empty_matches = all_matches.filter(length("Final_Flag") > 0)

    # The joins introduced 'None' values in some match/index columns; replace them with empty lists
    cols_replace_null = [col.name for col in drop_empty_matches.schema.fields if 'matches' in col.name or 'indices' in col.name]
    
    for col in cols_replace_null:
        drop_empty_matches = drop_empty_matches.withColumn(col, coalesce(col, array()))



    return drop_empty_matches.drop('Final_Flag')#.distinct() # Should I keep a 'distinct' here or would that risk data loss?
 


 


