export function render({ model, el }) {    
    let filePath = model.get("filePath");    
    el.classList.add("tweet-display");
    model.on("change:value", displayVals);
    let val = model.get("value");
    // if(val == null || val.length < 1){
    //     let box = document.createElement("div");
    //     box.classList.add("no-results");
    //     el.appendChild(box);
    // }
    // else{
        displayVals();
    // }

    function displayVals(){
        el.textContent = "";
        let value = model.get("value");
        for(let i = 0; i < value.length; i++){
            let row = JSON.parse(value[i]);
            let tweetBox = createAndAdd(el, "", "tweet-data");
            createAndAdd(tweetBox, "@" + row.SenderScreenName, "userName");
            createAndAdd(tweetBox, row.CreatedTime.substring(0, row.CreatedTime.indexOf(' ')), "date");
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