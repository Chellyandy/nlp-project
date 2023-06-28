"""
A module for obtaining repo readme and language data from the github API.
Before using this module, read through it, and follow the instructions marked
TODO.
After doing so, run it like this:
    python acquire.py
To create the `data.json` file that contains the data.
"""
import os
import json
from typing import Dict, List, Optional, Union, cast
import requests

from env import github_token, github_username

# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.

REPOS = ['octocat/Spoon-Knife',
 'github/gitignore',
 'Pierian-Data/Complete-Python-3-Bootcamp',
 'twbs/bootstrap',
 'rdpeng/ExData_Plotting1',
 'EbookFoundation/free-programming-books',
 'eugenp/tutorials',
 'torvalds/linux',
 'tensorflow/models',
 'TheOdinProject/css-exercises',
 'Yidadaa/ChatGPT-Next-Web',
 'DefinitelyTyped/DefinitelyTyped',
 'microsoft/vscode',
 'mdn/learning-area',
 'danielmiessler/SecLists',
 'soyHenry/Prep-Course',
 'rails/rails',
 'LarryMad/recipes',
 'apache/echarts',
 'keras-team/keras',
 'bitcoin/bitcoin',
 'qmk/qmk_firmware',
 'Yidadaa/ChatGPT-Next-Web',
 'DataScienceSpecialization/courses',
 'Significant-Gravitas/Auto-GPT',
 'django/django',
 'mdn/learning-area',
 'apache/dubbo',
 'hiifeng/V2ray-for-Doprax',
 'flutter/flutter',
 'yankils/hello-world',
 'rafaballerini/rafaballerini',
 'linuxacademy/devops-essentials-sample-app',
 'codebasics/py',
 'pandas-dev/pandas',
 'streamlit/streamlit-example',
 'netty/netty',
 'wesm/pydata-book',
 'soyHenry/fe-ct-prepcourse-fs',
 'jlevy/the-art-of-command-line',
 'mdn/content',
 'ColorlibHQ/AdminLTE',
 'ethereum/go-ethereum',
 'anuraghazra/github-readme-stats',
 'udacity/course-collaboration-travel-plans',
 'helm/charts',
 'AUTOMATIC1111/stable-diffusion-webui',
 'springframeworkguru/spring5webapp',
 'leereilly/swot',
 'gabrielecirulli/2048',
 'udacity/frontend-nanodegree-resume',
 'soyHenry/Prep-Course',
 'slatedocs/slate',
 '996icu/996.ICU',
 'othneildrew/Best-README-Template',
 'pjreddie/darknet',
 'rails/rails',
 'TheOdinProject/javascript-exercises',
 'jquery/jquery',
 'LarryMad/recipes',
 'angular/angular-cli',
 'Trinea/android-open-project',
 'Homebrew/homebrew-core',
 'kallaway/100-days-of-code',
 'adam-p/markdown-here',
 'spmallick/learnopencv',
 'fivethirtyeight/data',
 'avelino/awesome-go',
 'FFmpeg/FFmpeg',
 'apache/rocketmq',
 'kallaway/100-days-of-code',
 'adam-p/markdown-here',
 'fivethirtyeight/data',
 'rust-lang/rust',
 'grafana/grafana',
 'f/awesome-chatgpt-prompts',
 'woocommerce/woocommerce',
 'astaxie/build-web-application-with-golang',
 'xtekky/gpt4free',
 'Blankj/AndroidUtilCode',
 'OpenZeppelin/openzeppelin-contracts',
 'kallaway/100-days-of-code',
 'adam-p/markdown-here',
 'jhu-ep-coursera/fullstack-course4',
 'microsoft/Web-Dev-For-Beginners',
 'apachecn/ailearning',
 'fivethirtyeight/data',
 'rust-lang/rust',
 'necolas/normalize.css',
 'f/awesome-chatgpt-prompts',
 'ikatyang/emoji-cheat-sheet',
 'shadowsocks/shadowsocks-dotcloud',
 'Prince-Mendiratta/X-tra-Telegram',
 'leemunroe/responsive-html-email-template',
 'jenkinsci/docker',
 'gulpjs/gulp',
 'Koenkk/zigbee2mqtt.io',
 'electronicarts/CnC_Remastered_Collection',
 'careercup/CtCI-6th-Edition',
 'amueller/introduction_to_ml_with_python',
 'bloominstituteoftechnology/User-Interface',
 'ljpzzz/machinelearning',
 'wbond/package_control_channel',
 'learn-co-students/python-dictionaries-readme-data-science-intro-000',
 'oxford-cs-deepnlp-2017/lectures',
 'n8n-io/n8n',
 'ventoy/Ventoy',
 'udacity/frontend-nanodegree-arcade-game',
 'javascript-tutorial/en.javascript.info',
 'udacity/devops-intro-project',
 'waditu/tushare',
 'Uniswap/interface',
 'kubernetes/examples',
 'haha647/MagiskOnWSA-2',
 'johnpapa/angular-styleguide',
 'bradtraversy/vanillawebprojects',
 'learn-co-students/javascript-logging-lab-js-intro-000',
 'ironhack-labs/lab-react-ironcontacts',
 'carbon-design-system/carbon-tutorial',
 'android-async-http/android-async-http',
 'learn-co-students/javascript-logging-lab-bootcamp-prep-000',
 'prometheus-operator/prometheus-operator',
 'Z4nzu/hackingtool',
 'rabbitmq/rabbitmq-tutorials',
 'bloominstituteoftechnology/team-builder',
 'qier222/YesPlayMusic',
 'CoderMJLee/MJRefresh',
 'shadowsocks/shadowsocks-iOS',
 'ElemeFE/mint-ui',
 'mdbootstrap/mdb-ui-kit',
 'clearw5/Auto.js',
 'android/compose-samples',
 'square/leakcanary',
 'DeborahK/Angular-GettingStarted',
 'prettier/prettier',
 'purplecabbage/phonegap-plugins',
 'ironhack-labs/lab-html-exercise',
 'dotnet/roslyn',
 'IdentityServer/IdentityServer4',
 'rabbitmq/rabbitmq-server',
 'gunthercox/ChatterBot',
 'NLP-LOVE/ML-NLP',
 'emberjs/ember.js',
 'amueller/introduction_to_ml_with_python',
 'souravkl11/raganork-md',
 'microsoft/Windows-AppConsult-PWALab',
 'mckaywrigley/chatbot-ui',
 'jikexueyuanwiki/tensorflow-zh',
 'agalwood/Motrix',
 'enaqx/awesome-pentest',
 'Automattic/mongoose',
 'magenta/magenta',
 'aws/aws-cli',
 'apache/brpc',
 'iissnan/hexo-theme-next',
 'learn-co-students/js-beatles-loops-lab-bootcamp-prep-000',
 'Immortalin/onefraction',
 'ironhack-labs/lab-express-basic-auth',
 'learn-co-students/js-deli-counter-bootcamp-prep-000',
 'bloominstituteoftechnology/node-auth1-project',
 'doocs/leetcode',
 'grpc/grpc-go',
 'francistao/LearningNotes',
 'zhayujie/chatgpt-on-wechat',
 'baomidou/mybatis-plus',
 'TheAlgorithms/C',
 'dotnet/runtime',
 'kubernetes/dashboard',
 'learn-co-curriculum/react-hooks-components-basics-lab',
 'apache/dolphinscheduler',
 'learn-co-curriculum/phase-3-control-flow-conditional-statements',
 'TryGhost/Casper',
 'linuxacademy/cicd-pipeline-train-schedule-docker',
 'wix/react-native-calendars',
 'servo/servo',
 'hunkim/DeepLearningZeroToAll',
 'bloominstituteoftechnology/React-Router-Basic-Nav',
 'google/WebFundamentals',
 'edisga/nodejs-flatrisgame-devops',
 'termux/termux-packages']
 

headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )


def github_api_request(url: str) -> Union[List, Dict]:
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        return repo_info.get("language", None)
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_contents = requests.get(get_readme_download_url(contents)).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


def scrape_github_data() -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    return [process_repo(repo) for repo in REPOS]


if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data2.json", "w"), indent=1)