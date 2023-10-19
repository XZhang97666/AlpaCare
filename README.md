<p align="center" width="100%">
<img src="logo.png" style="width: 50%; min-width: 300px; display: block; margin: auto;">
</p>

# AlpaCare: Instruction-tuned Large Language Models for Medical Application

This is the repo for the AlpaCare, which are LLMs tuned on medical instrcutions. The repo contains:

- The 52K medcial instruction-response dataset used for fine-tuning AlpaCare.
- A 216 clinical craft free-form instrcution evulation test set.
- The weights of AlpaCare models (7B adn13B on LLaMA and LLaMA-2, respteively.)
- The code for
    1. medical task generation;
    2. fine-tunning LLaMA serises model;
    3. fine-tuned models response generation;
    4.  response evulation via LLMs.
 
## Overview
AlpaCare models contains 4 models (7B/13B - LLaMA[1]/LLaMA-2[2]) tunned on a 52k medical instruction-following data, following Alpaca[3] and Self-Instruct[4]   


[1]: LLaMA: Open and Efficient Foundation Language Models. Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample. https://arxiv.org/abs/2302.13971v1 

[2] Llama 2: Open foundation and fine-tuned chat models. Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. https://arxiv.org/abs/2307.09288

[3]: Stanford Alpaca: An Instruction-following LLaMA Model.Rohan Taori and Ishaan Gulrajani and Tianyi Zhang and Yann Dubois and Xuechen Li and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto. https://crfm.stanford.edu/2023/03/13/alpaca.html
[4]: Self-Instruct: Aligning Language Model with Self Generated Instructions. Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi. https://arxiv.org/abs/2212.10560


