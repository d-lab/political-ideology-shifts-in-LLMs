import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from umap import UMAP
from bertopic import BERTopic
import argparse
import json
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer



def create_topic_model(data: pd.DataFrame, text_column: str, num_topics: int, num_keywords: int, output_html_path: str = "topics.html"):
    print(f"1. Preparing to model {len(data)} documents into {num_topics} topics...")
    
    # Ensure the text column exists
    if text_column not in data.columns:
        raise ValueError(f"Column '{text_column}' not found in the DataFrame.")
        
    docs = data[text_column].tolist()

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    kmeans_model = KMeans(n_clusters=num_topics, random_state=42)
    vectorizer_model = CountVectorizer(stop_words="english")
    ctfidf_model = ClassTfidfTransformer()

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=kmeans_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        verbose=True
    )


    print("\n2. Fitting BERTopic model... (This may take a moment)")
    topics, _ = topic_model.fit_transform(docs)

    df_with_topics = data.copy()
    df_with_topics['topic'] = topics

    print(f"\n3. Extracting c-TF-IDF topic representations with top {num_keywords} keywords...")
    topic_representations_with_scores = topic_model.get_topics()

    # Create the dictionary with scores, but sliced to the number of keywords you want
    topic_representations_with_scores = {
        topic_id: keyword_tuples[:num_keywords]
        for topic_id, keyword_tuples in topic_representations_with_scores.items()
    }

    # Create the dictionary with only the words, also sliced
    topic_keywords_only = {
        topic_id: [word for word, score in keyword_tuples[:num_keywords]]
        for topic_id, keyword_tuples in topic_representations_with_scores.items()
    }

    # df_with_topics['topic_keywords'] = df_with_topics['topic'].map(topic_keywords_only)
    # df_with_topics['topic_keywords_with_scores'] = df_with_topics['topic'].map(topic_representations_with_scores)
    df_with_topics['topic_keywords'] = df_with_topics['topic'].map(topic_keywords_only)
    df_with_topics['topic_keywords_with_scores'] = df_with_topics['topic'].map(
        lambda x: json.dumps(topic_representations_with_scores.get(x, []))
    )


    print("\n4. Topic modeling complete.")
    print(f"Number of documents per topic:\n{df_with_topics['topic'].value_counts().sort_index()}")


    print(f"\n5. Generating interactive HTML visualization at '{output_html_path}'...")
    fig = topic_model.visualize_topics()
    fig.write_html(output_html_path)
    print("   Done.")
    
    return df_with_topics, topic_model


# ======================================= MAIN =======================================
def main():
    parser = argparse.ArgumentParser(description="Generate clusters based on persona descriptions.")
    parser.add_argument(
        '--clusters', 
        type=int,
        default=10,
        help="Number of cluster to make."
    )

    parser.add_argument(
        '--keys', 
        type=int,
        default=10,
        help="Number of keywords per topic."
    )

    args = parser.parse_args()

    N_CLUSTERS = args.clusters
    DATA_PATH = '../../data/processed/cleaned_persona.pqt'
    OUTPUT_DF_PATH = f'../../data/processed/clustered_persona/cleaned_persona_clustered_{N_CLUSTERS}topics.pqt'
    OUTPUT_HTML_PATH = f'../../data/processed/clustered_persona/political_{N_CLUSTERS}topics_visualization.html'
    TEXT_COLUMN_NAME = 'cleaned_persona'
    N_KEYWORDS_PER_TOPIC = args.keys

    main_df = pd.read_parquet(DATA_PATH)

    df_with_topics, fitted_model = create_topic_model(
        data=main_df,
        text_column=TEXT_COLUMN_NAME,
        num_topics=N_CLUSTERS,
        num_keywords=N_KEYWORDS_PER_TOPIC,
        output_html_path=OUTPUT_HTML_PATH
    )
    
    print("\n--- DataFrame with Topic Assignments ---")
    print(df_with_topics.head())

    # Save the dataframe with the new topic column
    df_with_topics.to_parquet(f"{OUTPUT_DF_PATH}", index=False)
    print(f"\nDataFrame with topics would be saved to '{OUTPUT_DF_PATH}'")
    
    print(f"\nProcess finished. Open '{OUTPUT_HTML_PATH}' in your browser to see the interactive results!")
# ======================================= MAIN =======================================


if __name__ == '__main__':
    main()