# AgentSims: An Open-Source Sandbox for Large Language Model Evaluation
How to evaluate the ability of large language models (LLM) is an open question after ChatGPT-like LLMs prevailing the community. Existing evaluation methods suffer from following shortcomings: (1) constrained evaluation abilities, (2) vulnerable benchmarks, (3) unobjective metrics. We suggest that task-based evaluation, where LLM agents complete tasks in a simulated environment, is a one-for-all solution to solve above problems. 

We present <a href="https://www.agentsims.com/" title="AgentSims">AgentSims</a>, an easy-to-use infrastructure for researchers from all disciplines to test the specific capacities they are interested in. Researchers can build their evaluation tasks by adding agents and buildings on an interactive GUI or deploy and test new support mechanisms, i.e. memory system and planning system, by a few lines of codes.  The demonstration is on https://agentsims.com/.

***Our system has better customization capabilities compared to other similar systems, as it is designed for open source custom task building. See our <a href="https://arxiv.org/abs/2308.04026" title="arXiv">arXiv paper</a>.*** 

![Image text](https://github.com/py499372727/AgentSims/blob/main/cover.png)

## Dependency
```
Python: 3.9.x
MySQL: 8.0.31
```
We recommend deploying on MacOS or Linux for better stability.

## API Key
For the security of your API Key, we have not included the parameter file in git. 

Copy the .env.example file to .env and fill in your API keys and configuration:
```
cp .env.example .env
```
Then edit the .env file with your settings:
```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-3.5-turbo  # or gpt-4o, etc.
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional, for OpenAI-compatible APIs
```

If you want to deploy your own model, see <a href="https://github.com/py499372727/AgentSims/wiki" title="DOCS">DOCS</a>.

## Development Notes

### LLM Integration
The project's LLM integration has been simplified. All LLM-related functionality is now located in `agent/utils/llm.py`. This module provides:

- `LLMCaller`: Main interface with JSON response parsing
- `OpenAICaller`: Base implementation for OpenAI compatible APIs
- `BaseCaller`: Simple base class for extending with other LLM implementations

When implementing new features, please use the classes from `agent/utils/llm` directly instead of the legacy implementations.

## Folder Creation
Before running, please
```
mkdir snapshot
mkdir logs
```
In addition, we recommend modifying the ***count_limit*** (number of loops per run) and ***cooldown*** (cooldown time between runs) in ***config/app.json*** to suitable values before running, in order to balance protection of your API key and experiment runtime efficiency.

If you encounter any issues during runtime, please first refer to our <a href="https://github.com/py499372727/AgentSims/wiki" title="DOCS">DOCS</a> in the wiki. If not resolved, please raise an issue or contact us directly.

--------------------------------------
To use our system, please follow these steps:

## 1.MySQL Init
MySQL is used for data storage on the server. After installing the appropriate version of MySQL, start the SQL service and initialize it as follows:
```
use mysql
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '';
flush privileges;

create database `llm_account` default character set utf8mb4 collate utf8mb4_unicode_ci;
create database `llm_game` default character set utf8mb4 collate utf8mb4_unicode_ci;
create database `llm_game0001` default character set utf8mb4 collate utf8mb4_unicode_ci;
create database `llm_game0002` default character set utf8mb4 collate utf8mb4_unicode_ci;
```

## 2.Install

```bash
pip install tornado
pip install mysql-connector-python
pip install websockets
pip install openai
```

or
```bash
pip install -r requirements.txt
```
## 3.Design Tasks
You can build tasks at this point. If you just want to try out the system first, you can skip this step. For task building, please refer to the <a href="https://github.com/py499372727/AgentSims/wiki" title="DOCS">DOCS</a> in the wiki or Section 4.2 Developer Mode in our <a href="https://arxiv.org/abs/2308.04026" title="arXiv">arXiv paper</a>.

## 4.Run Server

Start server:
```bash
./restart.sh
```
When you see
```
--------Server Started--------
```
in Server Terminal, the server has been started successfully.
## 5.Run Client
Once the server has started successfully, please launch the client. In the current version, we provide a web-based client. Please open ***client/index.html*** in your browser.

Note: Sometimes the web client fails to open correctly. We recommend right-clicking the ***index.html*** in your python IDE and select ***open in browser***. If you are familiar with ***nginx***, that is also a great choice. 

When you see
```
somebody linked.
```
in Server Terminal, the client has been started successfully.

## 6.Create agents and buildings
You can create agents and buildings at this point. For creation, please refer to the <a href="https://github.com/py499372727/AgentSims/wiki" title="DOCS">DOCS</a> in the wiki or Section 4.1 User Mode in our <a href="https://arxiv.org/abs/2308.04026" title="arXiv">arXiv paper</a>. 

## 7.Set Evaluation Target and Measurements
In AgentSims, evaluation are made by QA forms. Every k ticks, system would ask the subject agent an evaluation question. You can customize your subject agent, evaluation question, measurement of response in config/eval.json 
The example in config/eval.json shows an experiment called 'know pH'. The experiment will ask agent Alan 'Are you acquainted with pH' every 1 tick and if 'Yes' in response, the eval function will return True.
```
{
  "id": "know pH", # the human-readable name of evaluation, 
  "target_nickname": "Alan", # name of the subject agent
  "query": "Are you acquainted with pH ?", # evaluation qustion 
  "measurement": " 'Yes' in response" # measurement, 
  "interval": 1 # Evaluate every 1 tick
}

```

## 8.Run Simulation

You can start ***tick*** or ***mayor*** with the buttons on the web client. You can also
start with:
```bash
python -u tick.py
```
```bash
python -u mayor.py
```

For the difference with ***tick*** and ***mayor***, refer to our <a href="https://arxiv.org/abs/2308.04026" title="arXiv">arXiv paper</a>.

## 9.Restart
The following reset steps need to be performed each time upon restarting:
```
  rm -rf snapshot/app.json
```
```
  sudo mysql
  drop database llm_account;
  drop database llm_game0001;
  create database `llm_game0001` default character set utf8mb4 collate utf8mb4_unicode_ci;
  create database `llm_account` default character set utf8mb4 collate utf8mb4_unicode_ci;
```
```
 ./restart.sh
```
-------------------------------
## Authors and Citation
***Authors***: Jiaju Lin,<a href="https://twitter.com/zhaohao919041" title="twitter">Haoran Zhao</a>*,Aochi Zhang,Yiting Wu, Huqiuyue Ping,Qin Chen

***About Us***: PTA Studio is a startup company dedicated to providing a better open source architecture for the NLP community and more interesting AI games for players.

***Contact Us***: zhaohaoran@buaa.edu.cn

Please cite our paper if you use the code or data in this repository.
```
@misc{lin2023agentsims,
      title={AgentSims: An Open-Source Sandbox for Large Language Model Evaluation}, 
      author={Jiaju Lin and Haoran Zhao and Aochi Zhang and Yiting Wu and Huqiuyue Ping and Qin Chen},
      year={2023},
      eprint={2308.04026},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}