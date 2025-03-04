import warnings
warnings.filterwarnings('ignore')
import ipywidgets as widgets
import anywidget
import traitlets
import jupyter
from IPython.display import display, Javascript
import voila
from custom_widgets import *

TWEETS_PER_PAGE = 20
# JUPYTER_FILE_PATH = "../tree/images/"
JUPYTER_FILE_PATH = "images/"

class FilterModule():
    def displayPopUp(self, change=None):
        self.hasChanges = True
        self.filterBox.children = [self.mainFilters, self.popUpOptions]

    def __init__(self):
        self.filterBy = widgets.HTML(value = "Refine Results").add_class("heading5").add_class("medium")
        dateRange = widgets.HTML(value = "Date").add_class("body2").add_class("medium")
        self.fromDate = widgets.DatePicker(description = "From")
        self.toDate = widgets.DatePicker(description = "To")
        self.fromDate.add_class("date-constraint") # The script to set the elements attribute is attached to the toggleSwitch widget
        self.toDate.add_class("date-constraint") # This was done for convenience and should be changed later
        self.weightBy = WeightBy()
        self.dateBox = widgets.VBox([dateRange, widgets.HBox([self.fromDate, self.toDate])]).add_class("date-bar")
        self.allowRetweets = ToggleSwitch(label = "Include reposts")
        self.retweets = widgets.VBox([widgets.HTML(value = "Retweets").add_class("body2").add_class("medium"), self.allowRetweets])
        self.geography = SearchBar(header = "Geography", placeholder = "Search")
        self.userName = SearchBar(header = "Username", placeholder = "Search")
        self.mainPageClear = widgets.Button(description='Clear All').add_class("clear-button")
        self.mainPageSearch = widgets.Button(description='Apply').add_class("generic-button").add_class("main-page-search")
        self.popUpOptions = widgets.HBox([self.mainPageClear, self.mainPageSearch]).add_class("pop-up-options")
        self.userName.observe(self.displayPopUp, names=["value"])
        self.geography.observe(self.displayPopUp, names=["value"])
        self.allowRetweets.observe(self.displayPopUp, names=["value"])
        self.weightBy.observe(self.displayPopUp, names=["value"])
        self.fromDate.observe(self.displayPopUp, names=["value"])
        self.toDate.observe(self.displayPopUp, names=["value"])
        self.mainFilters = widgets.VBox([self.filterBy, self.dateBox, self.retweets, self.geography, self.userName, self.weightBy]).add_class("main-filters")
        self.filterBox = widgets.VBox([self.mainFilters]).add_class("filter-box")