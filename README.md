# Meme_Generation
Image-to-Text Model FineTuning for Meme Generation


To train and test this model, you must download the "Oxford_HIC" dataset first.
You can download it here https://github.com/runjiali-rl/Oxford_HIC/tree/master/data.

Here's an sample of this dataset below.
![image](https://github.com/user-attachments/assets/7193953c-7ed9-486e-bdb9-c3f9f67d1a7d)



We trained the CLIP-based model to generate humor caption for the given image like in Oxford-HIC dataset.

We fully trained the parameters of pre-trained CLIP model with Cross-Entropy Loss between the generated caption and the humor caption from the Oxford-HIC dataset.
Here are some examples of generated humor caption from our CLIP-based Finetuned model.

[1]
![스크린샷 2025-05-30 오후 3.37.43.png](attachment:bd22fd18-fa41-4009-aee8-e6e431ddc6ff:스크린샷_2025-05-30_오후_3.37.43.png)
![스크린샷 2025-05-30 오후 3.37.57.png](attachment:9edbc331-d34b-4d25-8776-bff483e42807:스크린샷_2025-05-30_오후_3.37.57.png)


[2]
![bokete_0.jpg](attachment:925ea65c-d92d-4967-92aa-d4a542810885:bokete_0.jpg)
![image](https://github.com/user-attachments/assets/72ed13e2-a4a3-49fa-9d1c-057f97e052a5)
