# Meme_Generation
Image-to-Text Model FineTuning for Meme Generation


To train and test our codes on you device, you must download the "Oxford_HIC" dataset first.
You can download it here https://github.com/runjiali-rl/Oxford_HIC/tree/master/data.

Here's an sample of this dataset below.
![image](https://github.com/user-attachments/assets/7193953c-7ed9-486e-bdb9-c3f9f67d1a7d)



We trained the CLIP-based model to generate humor caption for the given image like in Oxford-HIC dataset.

We fully trained the parameters of pre-trained CLIP model with Cross-Entropy Loss between the generated caption and the humor caption from the Oxford-HIC dataset.
Here are some examples of generated humor caption from our CLIP-based Finetuned model.

[1]





![image](https://github.com/user-attachments/assets/6a16361c-40a8-4ea7-bffd-a8485f421b0a)
![image](https://github.com/user-attachments/assets/e832097e-8c20-4d36-882c-8bf79cd09c94)



[2]




![image](https://github.com/user-attachments/assets/ae0b5b16-72ee-413b-82a4-ccf7ef869f6f)
![image](https://github.com/user-attachments/assets/72ed13e2-a4a3-49fa-9d1c-057f97e052a5)
