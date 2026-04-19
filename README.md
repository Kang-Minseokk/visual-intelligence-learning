# visual-intelligence-learning
This Repo is for visual-intelligence-learning projects

## Short Describe 
  Hi there 👋, Our team is "International" team.
  This repo initialize 'visual-intelligence-learning(DAI3004)'
  Build the customized model to classify CIFAR100 both supervised and self-supervised method

## How to Run? (Method 1. train from the scratch)
  1. First of all, make python environment <br>
      `pip install -r requirements.txt`
      
  2. Next, make sure your current location is among first_project or second_project <br>

  3. Please checkout branch to "first-final-challenge"
  
  4. Finally, we can train and test CIFAR100 using command under.<br>
      `python3 train.py --config configs/international_final.yaml --output ./output/<output-directory-you-want>`

     ⭐️ config and output argument is mandatory. You can visualize the result and training curve with under command. <br>
      `tensorboard --logdir ./output/<output-directory-you-want>`
    
     ⭐️ we can change seed using command under. <br>
      `python3 train.py --config configs/international_final.yaml --output ./output/<output-directory-you-want> --seed 40`

     ⭐️ Selection of config file will be announced after you execute the training by our teammates. If you want to check correcteness of config file, please contact me. (pilot920@hanyang.ac.kr) <br>
       ✅ current best performence config file : international_final.yaml

     ⭐️ Please check the test acc and loss in Final Test checking area!
         <img width="1543" height="307" alt="image" src="https://github.com/user-attachments/assets/95d6da08-35e8-45e3-9cd4-f516009c7e99" />


## Do you need help?
  Please contact our team members below. <br>
  * (Team Leader) Minseok Kang <br>
    E-mail : pilot920@hanyang.ac.kr <br>

  * (Team Leader) Soyoon Kim <br>
    E-mail : (comming soon) <br>

  * (Team Member) Xiang Li <br>
    E-mail : (comming soon) <br>
       
  
