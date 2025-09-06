# Regression Classification Analysis
Practice with LLM by identifying a real-world problem in which regression or classification analysis can be used.

# Assignment Details

1. Problem Identification

    When browsing a social media platform, there are opportunities to leave comments below posts that people on the platform have made. Rarely on a post with smaller engagement, but on more often on posts with at least a decent size viewing, there can be comments that can be classified as "toxic" towards the poster. The rise of social media has exponentially increased the amount of these kind of comments that can potentially harm the mental health of others online. 

    The target variable for this problem would be to classify a comment as TOXIC or NON-TOXIC. The main independent variable would be the text of the comment itself, as the model would attempt to predict the nature of the comment (TOXIC or NON-TOXIC) based on its text content. 


2. Findings With Example Strings
    
    I presented the model with four strings: "This post is dumb", "I love your hair", "Your car is cool dude", "Your outfit sucks bruh." These strings are what I believe you would typically find on an average instagram post. The model was able to decipher between which comments are potentially TOXIC or NON-TOXIC using key words found within the text content. The first and last classified as TOXIC, while the middle two are NON-TOXIC.


3. Discovering the Bias of the Model

    Asking the model to classify these four strings "I have a muslim friend", "I have a christian friend", "I have a white friend", and "I have a black friend", produces jarring results. As the model will print the strings containing "muslim" and "black" as toxic comments, therefore showing its bias against "muslim" in favor of "christian" and against "black" in favor of "white." This is one example of the model's faults, showing that this is not a perfect checker for toxicity. 


4. How Does the LR Model Compare to the SVC Model?

    Using the same example strings, the SVC model had similar results to the LR model, classifying the same strings as TOXIC, with the same others being NON-TOXIC. The SVC model displayed the same bias as the LR Model as well. 