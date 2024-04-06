# %% ### parameters

import pickle
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
#set font size
sns.set_context("paper", font_scale=1.2)

import matplotlib.pyplot as plt

suswiki_str = "sustainability-methods-wiki"
wikipedia_str = "wikipedia"
KNOWLEDGE_BASES = [suswiki_str, wikipedia_str]

bge_str = "bge-large-en-v1.5"
gte_str = "gte-large"
uae_str = "UAE-Large-V1"
EMBEDDINGS = [bge_str, gte_str, uae_str]

faiss_str = "db_faiss"
chroma_str = "db_chroma"
VECTORSTORES = [faiss_str, chroma_str]

eucledian_str = "l2"
cosine_str = "cosine"
innerproduct_str = "ip"
INDEX_DISTANCES = [eucledian_str, cosine_str, innerproduct_str]

#%% trial algorithm using trimmed values  

### trim experiment using pseudo value
# QUESTION_DATASET = QUESTION_DATASET[:10]
# FOLDER_PATH ="experiments/ALL/trim/"
TOP_K = [1,2]
LLMS = ["llama2", "mistral", "gpt35"]
INDEX_DISTANCES = [eucledian_str, innerproduct_str, cosine_str]
VECTORSTORES = [faiss_str, chroma_str]
EMBEDDINGS = [bge_str, gte_str,]
KNOWLEDGE_BASES = [ suswiki_str, wikipedia_str]

GENERATE_FLAG = False # to generate the answer csv and json - use it mainly for trigerring gpt35
EVALUATE_FLAG = False # to generate ragas evaluation and save it in csv and json


#%% presents the results

# create a result dataframe
all_result_df = pd.DataFrame()

for knowledge_base in KNOWLEDGE_BASES:
    for embedding in EMBEDDINGS:
        for vector_store_name in VECTORSTORES:
            for index_distance in INDEX_DISTANCES:
                for k in TOP_K:
                    for language_model in LLMS:
                        FOLDER_PATH = f"experiments/ALL/{knowledge_base}/{embedding}/{vector_store_name}/{index_distance}/"
                        if vector_store_name == faiss_str:
                            vector_store_name_experiment_file = "FAISS"
                        elif vector_store_name == chroma_str:
                            vector_store_name_experiment_file = "Chroma"

                        # load evaluation result
                        try:
                            with open(FOLDER_PATH + f"{language_model}_{vector_store_name_experiment_file}_{k}_RagasEval.pkl", 'rb') as f:
                                result = pickle.load(f)

                            new_row = pd.DataFrame({
                                'KB': [knowledge_base],
                                'Embedding': [embedding],
                                'Vector Store': [vector_store_name],
                                'Index Distance': [index_distance],
                                'k Docs': [k],
                                'LLM': [language_model],
                                'Answer Relevancy': [round(result['answer_relevancy'], 3)],
                                'Faithfulness': [round(result['faithfulness'], 3)],
                                'Context Recall': [round(result['context_recall'], 3)],
                                'Context Precision': [round(result['context_precision'], 3)],
                                'Answer Correctness': [round(result['answer_correctness'], 3)]
                            }) # input of the DataFrame is a dictionary

                            # Define the mappings
                            mappings = {
                                'KB': {
                                    'sustainability-methods-wiki': 'Suswiki',
                                    'wikipedia': 'Wikipedia'
                                },
                                'Vector Store': {
                                    'db_faiss': 'FAISS',
                                    'db_chroma': 'Chroma'
                                },
                                'Embedding': {
                                    'bge-large-en-v1.5': 'BGE',
                                    'gte-large': 'GTE',
                                    'UAE-Large-V1': 'UAE'
                                },
                                'Index Distance': {
                                    'l2': 'Eucledian',
                                    'cosine': 'Cosine',
                                    'ip': 'Inner Product'
                                },
                                'LLM': {
                                    'gpt35': 'GPT-3.5',
                                    'mistral': 'Mistral',
                                    'llama2': 'Llama2'
                                }
                            }

                            # Apply the mappings
                            for column, mapping in mappings.items():
                                new_row[column] = new_row[column].map(mapping).fillna(new_row[column])

                            all_result_df = pd.concat([all_result_df, new_row])
                            all_result_df = all_result_df.reset_index(drop=True)

                        except FileNotFoundError:
                            print(f"File not found:wou {FOLDER_PATH + f'{language_model}_{vector_store_name_experiment_file}_{k}_RagasEval.pkl'}")



#%% arrange the columns to be: KB	Embedding	Vector Store	Index Distance	k Docs	LLM Context Recall	Context Precision	Faithfulness    Answer Relevancy	Answer Correctness
from scipy.stats import hmean

# add retriever harmonic mean = harmonic_mean('Context Recall', 'Context Precision')
all_result_df['Retriever HMean'] = all_result_df[['Context Recall', 'Context Precision']].apply(hmean, axis=1).round(3)

# add generator harmonic mean = harmonic_mean('Faithfulness', 'Answer Relevancy', 'Answer Correctness')
all_result_df['Generator HMean'] = all_result_df[['Faithfulness', 'Answer Relevancy', 'Answer Correctness']].apply(hmean, axis=1).round(3)

# add harmonic mean = harmonic_mean('Context Recall', 'Context Precision', 'Faithfulness', 'Answer Relevancy', 'Answer Correctness')
all_result_df['RAGAS HMean'] = all_result_df[['Context Recall', 'Context Precision', 'Faithfulness', 'Answer Relevancy', 'Answer Correctness']].apply(hmean, axis=1).round(3)

#%%
all_result_df 

#%%
# create a seaborne exploration analysis of the result
# The independent variables are: KB, Embedding, Vector Store, Index Distance, k Docs, LLM
# The dependent variables are: Retriever HMean, Generator HMean, RAGAS HMean (combination of the retriever and generator Hmean)

# create pairplot to see the distribution of the independent variables
all_result_df_trim = all_result_df[[ 'KB', 'Embedding', 'Vector Store', 'Index Distance',  'LLM', 'Retriever HMean', 'Generator HMean', 'RAGAS HMean']]
# sns.pairplot(all_result_df_trim, hue='KB',diag_kind='kde', kind='scatter', palette='husl', plot_kws={'alpha': 0.6})
# sns.PairGrid(all_result_df_trim, hue='KB', diag_sharey=False).map_lower(sns.kdeplot, alpha=0.6).map_upper(sns.scatterplot, alpha=0.6).map_diag(sns.kdeplot, lw=3, alpha=0.6)

#%%   ALL DATAPOINTS: DIFFERENTIATE KB: create a 2d plot where x axis is the generator hmean and y axis is the retriever hmean
plt.rcParams['figure.figsize'] = [7, 5]  # Width, height in inches
sns.scatterplot(data=all_result_df, x='Generator HMean', y='Retriever HMean', hue='KB', style='LLM', s=150, alpha=0.7, edgecolor='black', palette='husl').set_title('Cluster of different KBs based on Generator and Retriever HMean')

#%% ALL DATAPOINTS BOXPLOT - DIFFERENTIATE KB
# plt.rcParams['figure.figsize'] = [7, 5]  # Width, height in inches
KB_boxplot = sns.boxplot(data=all_result_df, x='RAGAS HMean', y='KB',  width=.5, palette='coolwarm')#
# add label of the mean values of each KB
means = all_result_df.groupby('KB')['RAGAS HMean'].mean()

# Add mean labels
for i in range(len(means)):
    KB_boxplot.text(means[i], i-0.3, f'{means[i]:.2f}', horizontalalignment='center', color='black', weight='light')

KB_boxplot.set_xlim([0.5, 1])
KB_boxplot.set_title('Boxplot of different KBs based on RAGAS HMean')
plt.show()
#%% CREATE BOX PLOT
# create a pivot table to show the mean of the retriever hmean and generator hmean

sns.boxplot(data=all_result_df, x='RAGAS HMean', y='LLM', hue='Embedding', width=.5, palette='coolwarm').set_xlim([0.5, 1])
# label the mean of each of the boxplot


# Add in points to show each observation
sns.swarmplot(all_result_df, x='RAGAS HMean', y='LLM', hue='Embedding', size=7, palette='husl', alpha=1, linewidth=0.5)



# %% MANUAL filtering
filter_k = 2
filter_index_dist = "Eucledian"
filter_vectorstore = "FAISS"
filter_embedding="BGE"

filtered_df = all_result_df[
    # (all_result_df['k Docs'] == filter_k) & 
    (all_result_df['Index Distance'] == filter_index_dist) & 
    (all_result_df['Vector Store'] == filter_vectorstore) & 
    (all_result_df['Embedding'] == filter_embedding)]
filtered_df

plt.rcParams['figure.figsize'] = [7, 5]  # Width, height in inches

# create a 2d plot where x axis is the generator hmean and y axis is the retriever hmean
sns.scatterplot(data=filtered_df, x='Generator HMean', y='Retriever HMean', hue='k Docs', style='LLM', alpha=0.6, s = 200, edgecolor='black')



# %%
# 1. Unfiltered 'k Docs'
filtered_df1 = all_result_df[
    # (all_result_df['k Docs'] == filter_k) & 
    (all_result_df['Index Distance'] == filter_index_dist) & 
    (all_result_df['Vector Store'] == filter_vectorstore) & 
    (all_result_df['Embedding'] == filter_embedding)]

# 2. Unfiltered 'Index Distance'
filtered_df2 = all_result_df[
    (all_result_df['k Docs'] == filter_k) & 
    # (all_result_df['Index Distance'] == filter_index_dist) & 
    (all_result_df['Vector Store'] == filter_vectorstore) & 
    (all_result_df['Embedding'] == filter_embedding)]

# 3. Unfiltered 'Vector Store'
filtered_df3 = all_result_df[
    (all_result_df['k Docs'] == filter_k) & 
    (all_result_df['Index Distance'] == filter_index_dist) & 
    # (all_result_df['Vector Store'] == filter_vectorstore) & 
    (all_result_df['Embedding'] == filter_embedding)]

# 4. Unfiltered 'Embedding'
filtered_df4 = all_result_df[
    (all_result_df['k Docs'] == filter_k) & 
    (all_result_df['Index Distance'] == filter_index_dist) & 
    (all_result_df['Vector Store'] == filter_vectorstore) 
    # & (all_result_df['Embedding'] == filter_embedding)
]

# create 4 scatterplots based on the 4 filtered dataframes with titles 'k Docs', 'Index Distance', 'Vector Store', 'Embedding'
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
sns.scatterplot(data=filtered_df1, x='Generator HMean', y='Retriever HMean', hue='k Docs', style='LLM', alpha=0.6, palette='coolwarm', s = 200, edgecolor='black', ax=axes[0, 0])
sns.scatterplot(data=filtered_df2, x='Generator HMean', y='Retriever HMean', hue='Index Distance', style='LLM', alpha=0.6, palette='husl', s = 200, edgecolor='black', ax=axes[0, 1])
sns.scatterplot(data=filtered_df3, x='Generator HMean', y='Retriever HMean', hue='Vector Store', style='LLM', alpha=0.6, palette='husl', s = 200, edgecolor='black', ax=axes[1, 0])
sns.scatterplot(data=filtered_df4, x='Generator HMean', y='Retriever HMean', hue='Embedding', style='LLM', alpha=0.6, palette='husl', s = 200, edgecolor='black', ax=axes[1, 1])

# Set titles
axes[0, 0].set_title('Comparing Top k Documents with different LLMs')
axes[0, 1].set_title('Comparing Index Distance with different LLMs')
axes[1, 0].set_title('Comparing Vector Store with different LLMs')
axes[1, 1].set_title('Comparing Embedding with different LLMs')

#%% SWARMPLOTS


plt.rcParams['figure.figsize'] = [7, 5]  # Width, height in inches
# sns.boxplot(data=filtered_df4, x='RAGAS HMean', y='LLM', hue='Embedding', width=.5, palette='coolwarm').set_xlim([0.5, 1])
# label the mean of each of the boxplot

# Add in points to show each observation
# sns.stripplot(filtered_df4, x='RAGAS HMean', y='LLM', hue='Embedding', size=7, palette='husl', alpha=1, linewidth=0.5)

# Add horizontal lines
unique_y = filtered_df1['LLM'].unique()
for y in unique_y:
    plt.axhline(y, color='gray', linestyle='--', alpha=0.6)

# create a new strip plot for filtered_df1
sns.swarmplot(filtered_df1, x='RAGAS HMean', y='LLM' ,hue='k Docs',size=15, alpha=1, linewidth=0.5).set_title('Comparing Top k Documents with different LLMs')
plt.show()
sns.swarmplot(filtered_df2, x='RAGAS HMean', y='LLM', hue='Index Distance', size=15, palette='husl', alpha=1, linewidth=0.5).set_title('Comparing Index Distance with different LLMs')
plt.show()
sns.swarmplot(filtered_df3, x='RAGAS HMean', y='LLM', hue='Vector Store', size=15, palette='husl', alpha=1, linewidth=0.5).set_title('Comparing Vector Store with different LLMs')
plt.show()
sns.swarmplot(filtered_df4, x='RAGAS HMean', y='LLM', hue='Embedding', size=15, palette='husl', alpha=1, linewidth=0.5).set_title('Comparing Embedding with different LLMs')

#%% use forloop to add horizontal line to everything

for df in [filtered_df1, filtered_df2, filtered_df3, filtered_df4]:
    hue_filter = ''
    palette_col = 'husl'
    # if filtered_df1 -> then dont use hue=

    if filtered_df1 is df:
        hue_filter = 'k Docs'
        ax = sns.swarmplot(data=df, x='RAGAS HMean', y='LLM', hue=hue_filter, size=15, alpha=1, linewidth=0.5)
        ax.set_title(f'Comparing {hue_filter} with different LLMs')

        for y in unique_y:
            plt.axhline(y, color='gray', linestyle='--', alpha=0.6)
        plt.show()
        continue
    elif filtered_df2 is df:
        hue_filter = 'Index Distance'
        palette_col = 'husl'
    elif filtered_df3 is df:
        hue_filter = 'Vector Store'
        palette_col = 'husl'
    elif filtered_df4 is df:
        hue_filter = 'Embedding'
        palette_col = 'husl'

        

    ax = sns.swarmplot(data=df, x='RAGAS HMean', y='LLM', hue=hue_filter, size=15, palette='husl', alpha=1, linewidth=0.5)
    ax.set_title(f'Comparing {hue_filter} with different LLMs')

    for y in unique_y:
        plt.axhline(y, color='gray', linestyle='--', alpha=0.6)

    plt.show()


#%% SCATTERPLOT 
sns.scatterplot(filtered_df1, x='RAGAS HMean', y='LLM' ,hue='k Docs', style='LLM' , alpha=0.75, s=200, edgecolor='black').set_title('Comparing Top k Documents with different LLMs')
plt.show()
sns.scatterplot(filtered_df2, x='RAGAS HMean', y='LLM', hue='Index Distance', style='LLM', palette='husl', alpha=0.75, s=200, edgecolor='black').set_title('Comparing Index Distance with different LLMs')
plt.show()
sns.scatterplot(filtered_df3, x='RAGAS HMean', y='LLM', hue='Vector Store', style='LLM', palette='husl', alpha=0.75, s=200, edgecolor='black').set_title('Comparing Vector Store with different LLMs')
plt.show()
sns.scatterplot(filtered_df4, x='RAGAS HMean', y='LLM', hue='Embedding', style='LLM',palette='husl', alpha=0.75, s=200, edgecolor='black').set_title('Comparing Embedding with different LLMs')

#%% create a horizontal grouped bar chart to show the mean of the RAGAS HMean for each Index Distance

# create a pivot table to show the mean of the RAGAS HMean for each Index Distance
index_distance_pivot = filtered_df2.pivot_table(index='LLM', columns='Index Distance', values='RAGAS HMean')

# create a horizontal grouped bar chart
# hue is the index of the pivot table
sns.barplot(data=index_distance_pivot, hue=index_distance_pivot.index ,orient='h').set_title('Mean of the RAGAS HMean for each Index Distance')

#%% more pythonic way

# Define filters
filters = {
    'k Docs': 2,
    'Index Distance': "Eucledian",
    'Vector Store': "FAISS",
    'Embedding': "BGE"
}

# Create filtered dataframes
filtered_dfs = {f: all_result_df[all_result_df[col] == val] if col in filters else all_result_df for f, (col, val) in enumerate(filters.items())}

# Set plot size temporarily
with plt.rc_context({'figure.figsize': [7, 5]}):
    # Create a 2d plot where x axis is the generator hmean and y axis is the retriever hmean
    sns.scatterplot(data=filtered_dfs[0], x='Generator HMean', y='Retriever HMean', hue='k Docs', style='LLM', alpha=0.6, s = 200, edgecolor='black')
# %%
