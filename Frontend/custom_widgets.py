import warnings
warnings.filterwarnings('ignore')
import ipywidgets as widgets
import anywidget
import traitlets
import jupyter
from IPython.display import display, Javascript
import voila

TWEETS_PER_PAGE = 20
DEBUG_MODE = True
# JUPYTER_FILE_PATH = "../tree/images/"
JUPYTER_FILE_PATH = "images/"

class DummyElement(anywidget.AnyWidget):
    _esm = "anywidget/dummyscript.js"
    _css = "anywidget/misc.css"
    fileName = traitlets.Unicode().tag(sync=True)
    filePath = traitlets.Unicode(JUPYTER_FILE_PATH).tag(sync=True)
    size = traitlets.Int(0).tag(sync=True)

    alertTrigger = traitlets.Int(0).tag(sync=True)
    userResponse = traitlets.Int(2).tag(sync=True)
    changeSignal = traitlets.Int(0).tag(sync=True)

    calendarStart = traitlets.Unicode("").tag(sync=True)
    calendarEnd = traitlets.Unicode("").tag(sync=True)

    activeStanceAnalysis = traitlets.Int(0).tag(sync=True)

class SearchBar(anywidget.AnyWidget):
    _esm = "anywidget/searchBar.js"
    _css = "anywidget/searchBar.css"
    value = traitlets.Unicode("[]").tag(sync=True)
    header = traitlets.Unicode("").tag(sync=True)
    header2 = traitlets.Unicode("").tag(sync=True)
    placeholder = traitlets.Unicode("").tag(sync=True)
    count = traitlets.Int(0).tag(sync=True)
    
class TweetDisplay(anywidget.AnyWidget):
    _esm = "anywidget/tweetDisplay.js"
    _css = "anywidget/tweetDisplay.css"
    value = traitlets.List([]).tag(sync=True)
    keywords = traitlets.Unicode("").tag(sync=True)
    height = traitlets.Unicode("40vh").tag(sync=True)
    pageNum = traitlets.Int(1).tag(sync=True)
    maxPage = traitlets.Int(1).tag(sync=True)
    tweetsPerPage = traitlets.Int(TWEETS_PER_PAGE).tag(sync=True)
    filePath = traitlets.Unicode(JUPYTER_FILE_PATH).tag(sync=True)
    displayAddOn = traitlets.Int(0).tag(sync=True)
    addOnColumnName = traitlets.Unicode("").tag(sync=True)
    colorCode = traitlets.Int(0).tag(sync=True)
    stances = traitlets.List([]).tag(sync=True)
    stanceCorrection = traitlets.Unicode("").tag(sync=True)
    newStanceCorrectionNum = traitlets.Int(-2).tag(sync=True)

# class DatasetDisplay(anywidget.AnyWidget):
#     _esm = "anywidget/datasetDisplay.js"
#     _css = "anywidget/datasetDisplay.css"
#     size = traitlets.Int().tag(sync=True)
#     fileName = traitlets.Unicode().tag(sync=True)
#     filePath = traitlets.Unicode(JUPYTER_FILE_PATH).tag(sync=True)
    
class PageSelect(anywidget.AnyWidget):
    _esm = "anywidget/pageSelect.js"
    _css = "anywidget/pageSelect.css"
    value = traitlets.CInt(1).tag(sync=True)
    maxPage = traitlets.CInt(1).tag(sync=True)
    changeSignal = traitlets.Int(0).tag(sync=True)
    filePath = traitlets.Unicode(JUPYTER_FILE_PATH).tag(sync=True)

class WeightBy(anywidget.AnyWidget):
    _esm = "anywidget/weightBy.js"
    _css = "anywidget/weightBy.css"
    value = traitlets.Unicode("None").tag(sync=True)

class SortBar(anywidget.AnyWidget):
    _esm = "anywidget/sortBar.js"
    _css = "anywidget/sortBar.css"
    sortScope = traitlets.Unicode("Displayed Examples").tag(sync=True)
    columns = traitlets.List(["None", "Date", "Geography", "Retweets", "Username"]).tag(sync=True)
    columnNames = traitlets.List(["None", "CreatedTime", "State", "Retweets", "SenderScreenName"]).tag(sync=True)
    sortColumn = traitlets.Unicode("None").tag(sync=True)
    sortOrder = traitlets.Unicode("DESC").tag(sync=True)
    filePath = traitlets.Unicode(JUPYTER_FILE_PATH).tag(sync=True)

class SampleSelector(anywidget.AnyWidget):
    _esm = "anywidget/sampleSelector.js"
    _css = "anywidget/sampleSelector.css"
    filePath = traitlets.Unicode(JUPYTER_FILE_PATH).tag(sync=True)
    value = traitlets.Int(50).tag(sync=True)
    total = traitlets.Int(0).tag(sync=True)
    options = traitlets.List([50, 100, 150, 200]).tag(sync=True)
    label = traitlets.Unicode("").tag(sync=True)
    changeSignal = traitlets.Int(0).tag(sync=True)

class ToggleSwitch(anywidget.AnyWidget):
    _esm = "anywidget/toggleSwitch.js"
    _css = "anywidget/toggleSwitch.css"
    value = traitlets.Int(2).tag(sync=True)
    label = traitlets.Unicode("").tag(sync=True)
    hidden = traitlets.Int(0).tag(sync=True)

class ParameterDisplay(anywidget.AnyWidget):
    _esm = "anywidget/parameterDisplay.js"
    _css = "anywidget/parameterDisplay.css"
    headers = traitlets.List().tag(sync=True)
    value = traitlets.List().tag(sync=True)
    notFound = traitlets.Unicode().tag(sync=True)
    firstWord = traitlets.Unicode().tag(sync=True)
    secondWord = traitlets.Unicode().tag(sync=True)
    filePath = traitlets.Unicode(JUPYTER_FILE_PATH).tag(sync=True)


class AiSummary(anywidget.AnyWidget):
    _esm = "anywidget/aiSummary.js"
    _css = "anywidget/aiSummary.css"
    value = traitlets.List().tag(sync=True)
    rerender = traitlets.Int(0).tag(sync=True)
    sentenceNums = []
    unused = []
    selected = traitlets.CInt(0).tag(sync=True)
    changeSignal = traitlets.Int(0).tag(sync=True)

class LoadingPage(anywidget.AnyWidget):
    _esm = "anywidget/loadingScreen.js"
    _css = "anywidget/loadingScreen.css"
    filePath = traitlets.Unicode(JUPYTER_FILE_PATH).tag(sync=True)
    text = traitlets.Unicode().tag(sync=True)
    processRate = traitlets.Int(500).tag(sync=True)
    processInitial = traitlets.Int(-1).tag(sync=True)


class SemanticSearch(anywidget.AnyWidget):
    _esm = "anywidget/semanticSearch.js"
    _css = "anywidget/semanticSearch.css"
    value = traitlets.Unicode("").tag(sync=True)
    placeholder = traitlets.Unicode("").tag(sync=True)
    filterPercent = traitlets.CInt(50).tag(sync=True)

class StanceAnalysis(anywidget.AnyWidget):
    _esm = "anywidget/stanceAnalysis.js"
    _css = "anywidget/stanceAnalysis.css"
    pageNumber = traitlets.Int(0).tag(sync=True)
    topic = traitlets.Unicode("").tag(sync=True)
    stances = traitlets.List(["", "", "", ""]).tag(sync=True)
    examples = traitlets.List(["", "", "", ""]).tag(sync=True)
    colors = traitlets.List(["#EDF774", "#94E7DD", "#B8B7EE", "#D78DE9"]).tag(sync=True)
    filePath = traitlets.Unicode(JUPYTER_FILE_PATH).tag(sync=True)
    seenInfo = traitlets.Int(0).tag(sync=True)
    changeSignal = traitlets.Int(0).tag(sync=True)