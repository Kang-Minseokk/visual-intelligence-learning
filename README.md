# visual-intelligence-learning
This Repo is for visual-intelligence-learning projects

## Short Describe 
  Hi there 👋, Our team is "International" team.
  This repo initialize 'visual-intelligence-learning(DAI3004)'
  Build the customized model to classify CIFAR100 both supervised and self-supervised method

## How to Run? (Method 1. train from the scratch)
  1. First of all, make conda environment
      `conda env create -n <environment-name-you-want> python=3.9 -y`
     Additionally, match dependencies with our environment
      `conda env create -f environment.yml`
      
  3. Next, make sure your current location is among first_project or second_project
  
  4. Finally, we can train and test CIFAR100 using command under.
      `python train.py --config configs/base_config.yaml --output ./output/<output-directory-you-want>`

     ⭐️ config and output argument is mandatory. You can visualize the result and training curve with under command.
      `tensorboard --logdir ./output/<output-directory-you-want>`

     ⭐️ Selection of config file will be announced after you execute the training by our teammates. If you want to check correcteness of config file, please contact me. (pilot920@hanyang.ac.kr)
       current best performence config file : base_config.yaml

## Do you need help?
  Please contact our team members below.
  * (Team Leader) Minseok Kang 
    E-mail : pilot920@hanyang.ac.kr

  * (Team Leader) Soyoon Kim
    E-mail : (comming soon)

  * (Team Member) Xiang Li
    E-mail : (comming soon)
       
  
