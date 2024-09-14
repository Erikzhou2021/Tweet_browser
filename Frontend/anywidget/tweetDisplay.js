export function render({ model, el }) {    
    let filePath = model.get("filePath");    
    el.classList.add("tweet-display");
    let height = model.get("height");
    // el.setAttribute("style", "height: " + height);
    model.on("change:value", displayVals);
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
            let tweetBox = createAndAdd(el, "", "tweet-data");
            createAndAdd(tweetBox, "@" + row.SenderScreenName, "userName");
            let date = new Date(row.CreatedTime);
            let dateString = date.toISOString().split('T')[0];
            createAndAdd(tweetBox, dateString, "date");
            createAndAdd(tweetBox, '<img src= \"' + filePath + 'location.svg\" class="icon"> ' + makeNotNull(row.State, "Unknown"), "state");
            createAndAdd(tweetBox, '<img src= \"' + filePath + 'retweet.svg\" class="icon"> ' + makeNotNull(row.Retweets), "retweets");
            createAndAdd(tweetBox, '<img src= \"' + filePath + 'like.svg\" class="icon"> ' + makeNotNull(row.Favorites), "likes");
            createAndAdd(tweetBox, row.Message, "message");
            el.appendChild(tweetBox);
            if(i == middle){
                middleHeight = tweetBox.offsetTop;
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
        else if(Math.ceil(el.scrollTop + el.offsetHeight) >= el.scrollHeight && pageNum < model.get("maxPage")){
            pageNum++;
            updateFlag = 2;
        }
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
}