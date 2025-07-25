import sqlite3

# connect to sqlite database (or create if iit does not exist)


conn = sqlite3.connect("/Users/jeffreywongqiyuan/Desktop/cleaned_hea_v3/feedback.db")
# create a cursor
cursor = conn.cursor()

# create a table for storing data
cursor.execute("""
CREATE TABLE IF NOT EXISTS feedback(
               id INTEGER PRIMARY KEY AUTOINCREMENT, 
               ease_to_use_response TEXT, 
               performance_response TEXT, 
               prediction_accuracy_response TEXT, 
               feature_set_response TEXT, 
               suggestions_improvements_response TEXT)
""")

conn.commit()
conn.close()