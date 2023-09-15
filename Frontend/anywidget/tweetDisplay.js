export function render({ model, el }) {        
    el.classList.add("tweet-display");
    model.on("change:value", displayVals);

    displayVals();

    function displayVals(){
        el.textContent = "";
        let value = model.get("value");
        for(let i = 0; i < value.length; i++){
            let row = JSON.parse(value[i]);
            let tweetBox = createAndAdd(el, "", "tweet-data");
            createAndAdd(tweetBox, "@" + row.SenderScreenName, "userName");
            createAndAdd(tweetBox, row.CreatedTime.substring(0, row.CreatedTime.indexOf(' ')), "date");
            createAndAdd(tweetBox, '<img src="images/retweet.svg" class="icon"> ' + makeNotNull(row.Retweets), "retweets");
            createAndAdd(tweetBox, '<img src="images/like.svg" class="icon"> ' + makeNotNull(row.Favorites), "likes");
            createAndAdd(tweetBox, row.Message, "message");
            el.appendChild(tweetBox);
        }
    }
    function makeNotNull(val){
        if (val == null){
            return 0;
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