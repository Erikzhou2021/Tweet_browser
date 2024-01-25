export function render({ model, el }) {    
    let filePath = model.get("filePath");    
    el.classList.add("tweet-display");
    let height = model.get("height")
    el.style.setProperty('--height', height);
    model.on("change:value", displayVals);
    displayVals();

    function displayVals(){
        el.textContent = "";
        let value = model.get("value");
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
        }
    }
    function makeNotNull(val, replace = 0){
        if (val == null){
            return replace;
        }
        return val;
    }
    function createAndAdd(parent, html, cssClass){
        let temp = document.createElement("div");
        temp.innerHTML = html;
        temp.classList.add(cssClass);
        parent.appendChild(temp);
        return temp;
    }
    
}