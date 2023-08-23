export function render({ model, el }) {    
    el.classList.add("sort-bar");
    let sort = document.createElement("div");
    let temp = document.createElement("div");
    let sorted = document.createElement("input");
    sorted.setAttribute("type", "checkbox");
    sorted.classList.add("checkbox-round");
    temp.appendChild(sorted);
    

    let dropDowns = document.createElement("div");
    let label = document.createElement("strong");
    label.innerHTML = "&nbsp;Sort";
    sort.appendChild(temp);
    sort.appendChild(label);
    sort.classList.add("box-container");


    let dropDown1 = document.createElement("select");
    addOption(dropDown1, "Displayed Examples");
    addOption(dropDown1, "Entire Dataset");

    let byText = document.createElement("strong");
    byText.innerHTML = "&nbsp; By &nbsp;";

    let dropDown2 = document.createElement("select");
    addOption(dropDown2, "Date");
    addOption(dropDown2, "Geography");
    addOption(dropDown2, "Retweets");
    addOption(dropDown2, "Username");

    dropDowns.appendChild(dropDown1);
    dropDowns.appendChild(byText);
    dropDowns.appendChild(dropDown2);
    dropDowns.classList.add("dropdowns");

    let orderBy = document.createElement("div");
    let asc = document.createElement("img");
    asc.src = "images/ascending.svg";

    let dsc = document.createElement("img");
    dsc.src = "images/descending.svg";
    orderBy.appendChild(asc);
    orderBy.appendChild(dsc);

    function addOption(parent, value, css = ""){
        let temp = document.createElement("option");
        temp.value = value;
        temp.innerHTML = value;
        if(css != ""){
            temp.classList.add(css);
        }
        parent.appendChild(temp);
    }

    el.appendChild(sort);
    el.appendChild(dropDowns);
    el.appendChild(orderBy);
}