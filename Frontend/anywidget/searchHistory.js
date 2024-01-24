export function render({ model, el }) { 
    let filePath = model.get("filePath");  
    el.classList.add("search-history");  

    let left = document.createElement("img");
    left.src = filePath + "left.svg";
    left.classList.add("arrow");
    let leftText = document.createElement("span");
    leftText.innerHTML = "Prev";

    let middle = document.createElement("div");
    middle.innerHTML = "1";
    middle.classList.add("middle");

    let right = document.createElement("img");
    right.src = filePath + "right.svg";
    right.classList.add("arrow");
    let rightText = document.createElement("span");
    rightText.innerHTML = "Next";

    el.appendChild(left);
    el.appendChild(leftText);
    el.appendChild(middle);
    el.appendChild(rightText);
    el.appendChild(right);
}