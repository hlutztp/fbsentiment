import streamlit as st
from textblob import TextBlob
from collections import Counter
import re
import nltk

# Download NLTK stopwords if not already downloaded


def main():
    """
    Main function to run the Streamlit application for Facebook sentiment analysis.
    """
    #title/subheader set here
    st.title("Facebook Group Sentiment Analysis")
    st.subheader("Enter weekly posts from a Facebook Coach Group")
    
    #input from users is set here
    facebook_comments = st.text_area("Enter Facebook posts here, leave a line break between posts")

    # Custom lexicon for sentiment analysis
    custom_lexicon = {
        # Positive words
        'love': 2.0, 'fantastic': 2.0, 'helpful': 1.5, 'amazing': 2.0, 'great': 1.5, 'awesome': 1.5,
        'appreciate': 1.5, 'enjoy': 1.5, 'easy': 1.5, 'intuitive': 1.5, 'user-friendly': 1.5,
        'efficient': 1.5, 'streamlined': 1.5, 'well-designed': 1.5, 'smooth': 1.5, 'seamless': 1.5,
        'responsive': 1.5, 'accurate': 1.5, 'clear': 1.5, 'collaborative': 1.5, 'supportive': 1.5,
        'connected': 1.5, 'helpful feature': 1.5, 'successful': 1.5, 'benefit': 1.5, 'reliable': 1.5,
        'improving': 1.5, 'motivating': 1.5, 'insightful': 1.5, 'productive': 1.5, 'consistent': 1.5,
        'solid': 1.5, 'well-organized': 1.5, 'excellent': 2.0, 'top-notch': 2.0, 'highly recommend': 2.0,
        'well-executed': 1.5, 'collaborative environment': 1.5, 'positive outcome': 1.5, 'empowering': 1.5,
        'excited': 1.5, 'motivating': 1.5, 'flexibility': 1.5, 'powerful': 1.5, 'clarity': 1.5, 'potential': 1.0,
        'eager': 1.0, 'looking forward': 1.5, 'curious': 1.0, 'could you help': 0.5, 'would it be possible': 0.5,
        'suggestion': 0.5,

        # Neutral words
        'data': 0.0, 'session': 0.0, 'training': 0.0, 'workout': 0.0, 'feedback': 0.0, 'effort': 0.0,
        'schedule': 0.0, 'goal': 0.0, 'performance': 0.0, 'tracking': 0.0, 'program': 0.0, 'function': 0.0,
        'feature': 0.0, 'metrics': 0.0, 'pace': 0.0,

        # Negative words
        'frustrating': -1.5, 'disappointing': -1.5, 'confusing': -1.5, 'clunky': -1.5, 'slow': -1.0,
        'unreliable': -1.5, 'inaccurate': -1.5, 'outdated': -1.5, 'broken': -2.0, 'difficult': -1.5,
        'complicated': -1.5, 'problem': -1.5, 'issue': -1.0, 'annoying': -1.5, 'not working': -1.5,
        'error': -1.5, 'glitchy': -1.5, 'needs improvement': -1.5, 'basic feature': -1.0, 'limited': -1.0,
        'unintuitive': -1.5, 'inconsistent': -1.5, 'lacks flexibility': -1.0, 'tedious': -1.0,
        'slow response': -1.0, 'poor design': -2.0, 'time-consuming': -1.5, 'unresponsive': -1.5,
        'difficult to navigate': -1.5, 'overly complex': -1.5, 'bugs': -1.5, 'unreliable sync': -1.5,
        'crashes': -2.0, 'not user-friendly': -1.5, 'poorly implemented': -1.5, 'overwhelming': -1.0,
        'needs work': -1.0
    }

    def analyze_sentiment(comments, lexicon):
        """
        Analyzes the sentiment of a list of text comments using TextBlob and a custom lexicon.
        
        Args:
            comments (list): A list of text strings (comments).
            lexicon (dict): A dictionary mapping words to sentiment scores.
        
        Returns:
            tuple: A tuple containing the average TextBlob polarity, TextBlob subjectivity,
                   and custom lexicon score.
        """
        if not comments:
            return 0, 0, 0
        
        total_polarity, total_subjectivity, total_custom_score = 0, 0, 0
        
        for comment in comments:
            # TextBlob analysis
            analysis = TextBlob(comment)
            total_polarity += analysis.sentiment.polarity
            total_subjectivity += analysis.sentiment.subjectivity
            
            # Custom lexicon analysis
            words = re.sub(r'[^\w\s]', '', comment.lower()).split()
            custom_score = 0
            
            i = 0
            while i < len(words):
                word = words[i]
                if word == 'not' and i + 1 < len(words) and words[i + 1] in lexicon:
                    custom_score -= lexicon[words[i + 1]]
                    i += 2
                elif word in lexicon:
                    custom_score += lexicon[word]
                    i += 1
                else:
                    i += 1
            total_custom_score += custom_score
            
        avg_polarity = total_polarity / len(comments)
        avg_subjectivity = total_subjectivity / len(comments)
        avg_custom_score = total_custom_score / len(comments)
        
        return avg_polarity, avg_subjectivity, avg_custom_score

    def calculate_combined_score(avg_polarity, avg_custom_score):
        """
        Calculates a combined sentiment score from TextBlob polarity and a custom score.
        
        Args:
            avg_polarity (float): The average polarity score from TextBlob.
            avg_custom_score (float): The average score from the custom lexicon.
        
        Returns:
            float: The combined score.
        """
        # We need to scale both scores to a consistent range before averaging.
        # TextBlob polarity ranges from -1 to 1.
        # Custom lexicon scores can vary; we need to normalize it.
        # Let's assume a normalization based on the custom lexicon values.
        # Since the lexicon has values between -2 and 2, a similar range is reasonable.
        
        # Scale the custom score to a -1 to 1 range (assuming a max possible score of 2)
        # This is a simple scaling; a more robust method might be needed for different lexicons.
        scaled_custom_score = avg_custom_score / 2.0
        
        # Average the two scores
        return (avg_polarity + scaled_custom_score) / 2

    # Split the text input by lines
    facebook_comments_list = facebook_comments.split('\n') if facebook_comments else []

    if st.button("Analyze Facebook Posts"):
        if not facebook_comments_list:
            st.warning("Please enter some Facebook posts to analyze.")
            return

        # Get average sentiment scores for Facebook group comments
        fb_avg_polarity, fb_avg_subjectivity, fb_avg_custom_score = analyze_sentiment(facebook_comments_list, custom_lexicon)

        # Calculate the combined score for Facebook posts
        combined_fb_score = calculate_combined_score(fb_avg_polarity, fb_avg_custom_score)

        # Let's define the final score on a scale from 1 to 10
        # The combined score currently ranges from approximately -1 to 1
        # To scale it to a 1-10 range: (score - min_score) / (max_score - min_score) * (new_max - new_min) + new_min
        # Assuming our combined score ranges from -1 to 1, this gives:
        final_fb_score = ((combined_fb_score + 1) / 2) * 9 + 1
        
        # Display the output
        st.subheader("Results")
        st.write(f"**Average TextBlob Polarity:** {fb_avg_polarity:.2f}")
        st.write(f"**Average Custom Lexicon Score:** {fb_avg_custom_score:.2f}")
        st.write(f"**Combined Score (1-10 scale):** {final_fb_score:.2f}")
        
if __name__ == "__main__":
    main()
