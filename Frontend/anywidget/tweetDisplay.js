export function render({ model, el }) {   
    // let height = model.get("height");
    var rect = el.getBoundingClientRect();
    // let height = window.innerHeight - rect.top;
    // el.setAttribute("style", "max-height: " + height); 
    let filePath = model.get("filePath");    
    el.classList.add("tweet-display");
    model.on("change:value", displayVals);
    model.on("change:displayAddOn", displayVals)
    let pageNum = 1;
    el.onscroll = getNewTweets;
    let middle = model.get("tweetsPerPage");
    let updateFlag = 0; // 0 = no update, 1 = waiting for prev page, 2 = waiting for next page
    displayVals();

    function displayVals(){
        pageNum = model.get("pageNum");
        el.textContent = "";
        let value = model.get("value");
        let middleHeight = 0;
        if(value == undefined || value.length < 1){
            let box = document.createElement("div");
            box.classList.add("no-results");
            let title = document.createElement("h1");
            title.innerHTML = "No Results Found";
            let text = document.createElement("div");
            text.innerHTML = "Click <b>MODIFY SEARCH</b> to change keywords or filters";
            box.appendChild(title);
            box.appendChild(text);
            el.appendChild(box);
            return;
        }
        for(let i = 0; i < value.length; i++){
            let row = JSON.parse(value[i]);
            let tweetContainer = createAndAdd(el, "", "tweet-container");
            if(model.get("colorCode") > 0){
                let colorBox = createAndAdd(tweetContainer, "", "color-code");
                colorBox.classList.add("color" + row.stance);
            }
            let tweetBox = createAndAdd(tweetContainer, "", "tweet-data");
            createAndAdd(tweetBox, "@" + row.SenderScreenName, "userName");
            let date = new Date(row.CreatedTime);
            let dateString = date.toISOString().split('T')[0];
            createAndAdd(tweetBox, '<img src= \"' + filePath + 'calendar.svg\" class="icon"> ' + dateString, "date");
            createAndAdd(tweetBox, '<img src= \"' + filePath + 'location.svg\" class="icon"> ' + makeNotNull(row.State, "Unknown"), "state");
            createAndAdd(tweetBox, '<img src= \"' + filePath + 'retweet.svg\" class="icon"> ' + makeNotNull(row.Retweets), "retweets");
            createAndAdd(tweetBox, '<img src= \"' + filePath + 'like.svg\" class="icon"> ' + makeNotNull(row.Favorites), "likes");
            if(model.get("colorCode") > 0){
                makeOptionSelect(tweetBox, row.stance, row.Message);
            }
            let message = row.Message;
            let keywords = model.get("keywords");
            if(keywords != ""){
                let words = model.get("keywords").split(',');
                const regex = new RegExp(`\\b${words.join('|')}\\b`, 'gi');
                message = message.replaceAll(regex, "<b>$&</b>");
            }
            createAndAdd(tweetBox, message, "message");
            if(i == middle){
                middleHeight = tweetBox.offsetTop;
            }
            if(model.get("displayAddOn") > 0){
                createAndAdd(tweetContainer, row[model.get("addOnColumnName")], "add-on");
            }
        }
        if(pageNum > 1){
            if(updateFlag == 1){
                el.scrollTop = middleHeight - 4;
            }
            else if(updateFlag == 2){
                el.scrollTop = middleHeight - 4 - el.offsetHeight * 0.95;
            }else{
                el.scrollTop = 0;
            }
        }
        else{ // TODO: keep scroll top the same after a user closes out the advanced menu without searching
            el.scrollTop = 0;
        }
        updateFlag = 0;
    }

    function makeOptionSelect(tweetBox, currStanceNum, tweetText){
        let container = document.createElement("div");
        container.classList.add("dropdown");
        let button = document.createElement("button");
        button.innerHTML = "&#183;&#183;&#183;";
        button.classList.add("dropdownbtn");
        button.addEventListener("click", (event) => container.classList.toggle("open"));
        // button.addEventListener("blur", (event) => container.classList.remove("open"));
        let firstLayer = document.createElement("ul");
        firstLayer.classList.add("dropdown-menu");
        let currentStance = document.createElement("li");
        currentStance.innerHTML = "Current Stance";
        currentStance.classList.add("metatext");
        let currentStanceNumber = document.createElement("li");
        currentStanceNumber.innerHTML = "Stance " + String(parseInt(currStanceNum) + 1);
        if(currStanceNum == "-1"){
            currentStanceNumber.innerHTML = "Irrelavent";
        }
        currentStanceNumber.classList.add("color" + currStanceNum);
        let newStance = document.createElement("li");
        newStance.innerHTML = "New Stance";
        newStance.classList.add("metatext");

        tweetBox.appendChild(container);
        container.appendChild(button);
        container.appendChild(firstLayer);
        firstLayer.appendChild(currentStance);
        firstLayer.appendChild(currentStanceNumber);
        firstLayer.appendChild(newStance);
        let availableStances = model.get("stances");
        for(var i = 0; i < availableStances.length; i++){
            const currentStance = availableStances[i];
            if(currentStance != parseInt(currStanceNum)){
                let tempStance = document.createElement("li");
                tempStance.innerHTML = '<img src= \"' + filePath + 'option_select.svg\" class="icon"> ';
                if(currentStance == -1){
                    tempStance.innerHTML += "Irrelavent";
                }
                else{
                    tempStance.innerHTML += "Stance " + String(currentStance + 1);
                }
                tempStance.classList.add("color" + String(currentStance));
                tempStance.addEventListener("click", (event) => { 
                    model.set("stanceCorrection", tweetText);
                    model.set("newStanceCorrectionNum", currentStance);
                    model.save_changes();
                    closeDropdowns();
                    tempStance.parentElement.parentElement.parentElement.parentElement.classList.add("stance-corrected");
                    // alert(tempStance.parentElement.parentElement.parentElement.parentElement.matches('.tweet-container'));
                });
                firstLayer.appendChild(tempStance);
            }
        }
    }

    function makeNotNull(val, replace = 0){
        if (val == null){
            return replace;
        }
        return val;
    }
    function getNewTweets(){
        pageNum = model.get("pageNum");
        if(updateFlag != 0){
            return;
        }
        if (el.scrollTop == 0 && pageNum > 1){
            pageNum--;
            updateFlag = 1;
        } 
        else if(Math.abs(el.scrollHeight - el.clientHeight - el.scrollTop) <= 2 && pageNum < model.get("maxPage")){
            pageNum++;
            updateFlag = 2;
        }
        // Math.ceil(el.scrollTop + el.offsetHeight) >= el.scrollHeight
        model.set("pageNum", pageNum);
        model.save_changes();
    }
    function createAndAdd(parent, html, cssClass){
        let temp = document.createElement("div");
        temp.innerHTML = html;
        temp.classList.add(cssClass);
        parent.appendChild(temp);
        return temp;
    }
    function closeDropdowns(){
        var dropdowns = document.getElementsByClassName("dropdown");
        for (var i = 0; i < dropdowns.length; i++) {
            var openDropdown = dropdowns[i];
            if (openDropdown.classList.contains('open')) {
                openDropdown.classList.remove('open');
            }
        }
    }
    window.onclick = function(event) {
        if (!event.target.matches('.dropdown button') && !event.target.matches(".dropdown li")) {
            closeDropdowns();
        }
    }
}