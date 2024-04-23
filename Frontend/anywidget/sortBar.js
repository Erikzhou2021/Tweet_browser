export function render({ model, el }) { 
    let filePath = model.get("filePath");   
    el.classList.add("sort-bar");

    let dropDowns = document.createElement("div");
    let label = document.createElement("strong");
    label.innerHTML = "&nbsp;Sort By";

    let dropDown = document.createElement("select");
    addOption(dropDown, "None", "None");
    addOption(dropDown, "Date", "CreatedTime");
    addOption(dropDown, "Geography", "State");
    addOption(dropDown, "Retweets", "Retweets");
    addOption(dropDown, "Username", "SenderScreenName");
    dropDown.addEventListener("change", updateSortColumn);
    dropDown.value = model.get("sortColumn");

    function updateSortColumn(){
        let sortColumn = dropDown.value;
        model.set("sortColumn", sortColumn);
        model.save_changes();
    }

    dropDowns.appendChild(dropDown);
    dropDowns.classList.add("dropdowns");

    let orderBy = document.createElement("div");
    let arrow = document.createElement("img");
    arrow.src = filePath + "arrow_down.svg";
    arrow.addEventListener("click", updateOrder);

    function updateOrder(){
        if(model.get("sortOrder") == "ASC"){
            arrow.src = filePath + "arrow_down.svg";
            model.set("sortOrder", "DESC");
        }
        else{
            arrow.src = filePath + "arrow_up.svg";
            model.set("sortOrder", "ASC");
        }
        model.save_changes();
    }

    orderBy.appendChild(arrow);

    function addOption(parent, text, value, css = ""){
        let temp = document.createElement("option");
        temp.value = value;
        temp.innerHTML = text;
        if(css != ""){
            temp.classList.add(css);
        }
        parent.appendChild(temp);
    }

    el.appendChild(label);
    el.appendChild(dropDowns);
    el.appendChild(orderBy);
}