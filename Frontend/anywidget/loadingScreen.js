export function render({ model, el }) { 
    el.classList.add("loading-page");
    let spinner = document.createElement("div");
    spinner.classList.add("spinner");
    let text = document.createElement("div");
    text.classList.add("medium");
    text.classList.add("heading4");
    text.innerHTML = model.get("text");
    let i = 0;

    let timeLeft = document.createElement("div");
    timeLeft.classList.add("heading5");
    timeLeft.classList.add("medium");
    let tweetsLeft = model.get("processInitial");

    function calcTimeLeft(){
        let minsLeft = Math.round(tweetsLeft / model.get("processRate"));
        let timeText = "";
        
        if(minsLeft > 60){
            let hours = Math.floor(minsLeft / 60);
            if(hours == 1){
                timeText = "1 hour ";
            }
            else{
                timeText = hours.toString() + " hours ";
            }
        }
        if(minsLeft < 1){
            timeText = "less than 1 minute";
        }
        else{
            let mins = minsLeft % 60;
            if(minsLeft == 1){
                timeText += "1 minute";
            }
            else {
                timeText += mins.toString() + " minutes";
            }
        }
        timeLeft.innerHTML = "Estimated Time: " + timeText;
    }

    function updateTime(){
        tweetsLeft -= model.get("processRate");
        calcTimeLeft();
    }
    function addDots(){
        text.innerHTML = model.get("text") + Array(i+2).join(".");
        i++;
        i %= 3;
    }

    setInterval(addDots, 1000);

    el.appendChild(spinner);
    el.appendChild(text);
    if(tweetsLeft != -1){
        el.appendChild(timeLeft);
        calcTimeLeft();
        setInterval(updateTime, 60000);
    }
}