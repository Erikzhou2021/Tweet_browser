export function render({ model, el }) { 
    let MAX_PAGES_AROUND_CURRENT= 3;
    let count = model.get("count");
    let current = model.get("current");
    if(count == 0){
        return;
    }
    let filePath = model.get("filePath");  
    el.classList.add("search-history");  

    let left = document.createElement("img");
    left.src = filePath + "left.svg";
    left.classList.add("arrow");
    let leftText = document.createElement("span");
    leftText.innerHTML = "Prev";
    left.addEventListener("click", function(){ changeCurrent(current-1); });
    leftText.addEventListener("click", function(){ changeCurrent(current-1); });

    let middle = document.createElement("div");
    middle.classList.add("middle-nums");
    for(var i = Math.max(1, current - MAX_PAGES_AROUND_CURRENT); i < current; i++){
        let ellipse = document.createElement("img");
        ellipse.src = filePath + "ellipse.svg";
        ellipse.classList.add("middle-nums");
        let index = i;
        ellipse.addEventListener("click", function(){ changeCurrent(index); });
        middle.appendChild(ellipse);
    }
    let currPage = document.createElement("div");
    currPage.innerHTML = current;
    middle.appendChild(currPage);
    for(var i = current+1; i <= Math.min(count, current + MAX_PAGES_AROUND_CURRENT); i++){
        let ellipse = document.createElement("img");
        ellipse.src = filePath + "ellipse.svg";
        ellipse.classList.add("middle-nums");
        let index = i;
        ellipse.addEventListener("click", function(){ changeCurrent(index); });
        middle.appendChild(ellipse);
    }

    let right = document.createElement("img");
    right.src = filePath + "right.svg";
    right.classList.add("arrow");
    let rightText = document.createElement("span");
    rightText.innerHTML = "Next";
    right.addEventListener("click", function(){ changeCurrent(current+1); });
    rightText.addEventListener("click", function(){ changeCurrent(current+1); });

    function changeCurrent(val){
        if(val < 1 || val > count){
            return;
        }
        let changeSignal = model.get("changeSignal") + 1;
        model.set("current", val);
        model.set("changeSignal", changeSignal);
        model.save_changes();
    }

    el.appendChild(left);
    el.appendChild(leftText);
    el.appendChild(middle);
    el.appendChild(rightText);
    el.appendChild(right);
}