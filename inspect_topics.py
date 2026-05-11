from bertopic import BERTopic
import pandas as pd

# 1. Load the saved model
# Since we used BERTić, we tell BERTopic which embedding model to link to
model_path = "bertic_politics_model"
topic_model = BERTopic.load(model_path)

# 2. Get the topic information table
# This gives you: Topic ID, Count, and the Name (Top words)
topic_info = topic_model.get_topic_info()

# 3. Print the results nicely
print("\n" + "="*30)
print("📊 EXTRACTED POLITICS TOPICS")
print("="*30)

# We'll display the Top 10 words for the first 10 topics as a preview
# topic_info contains columns: Topic, Count, Name, Representation, etc.
print(topic_info[['Topic', 'Count', 'Name']].to_string(index=False))

# 4. Optional: Save to CSV to look at them in Excel/Calc
topic_info.to_csv("extracted_topics_report.csv", index=False)
print(f"\n✅ Detailed report saved to 'extracted_topics_report.csv'")

# If you want to see the specific words for a specific topic (e.g., Topic 0)
# print(topic_model.get_topic(0))