export function render({ model, el }) {    
    el.classList.add("sort-bar");
    let sort = document.createElement("div");
    let temp = document.createElement("div");

    let dropDowns = document.createElement("div");
    let label = document.createElement("strong");
    label.innerHTML = "&nbsp;Sort";
    sort.appendChild(temp);
    sort.appendChild(label);
    sort.classList.add("box-container");

    let dropDown1 = document.createElement("select");
    addOption(dropDown1, "Displayed Examples", "Displayed Examples");
    addOption(dropDown1, "Entire Dataset", "Entire Dataset");
    dropDown1.addEventListener("change", updateSortScope);
    dropDown1.value = model.get("sortScope");

    function updateSortScope(){
        let sortScope = dropDown1.value;
        model.set("sortScope", sortScope);
        model.save_changes();
    }

    let byText = document.createElement("strong");
    byText.innerHTML = "&nbsp; By &nbsp;";

    let dropDown2 = document.createElement("select");
    addOption(dropDown2, "None", "None");
    addOption(dropDown2, "Date", "CreatedTime");
    addOption(dropDown2, "Geography", "State");
    addOption(dropDown2, "Retweets", "Retweets");
    addOption(dropDown2, "Username", "SenderScreenName");
    dropDown2.addEventListener("change", updateSortColumn);
    dropDown2.value = model.get("sortColumn");

    function updateSortColumn(){
        let sortColumn = dropDown2.value;
        model.set("sortColumn", sortColumn);
        model.save_changes();
    }

    dropDowns.appendChild(dropDown1);
    dropDowns.appendChild(byText);
    dropDowns.appendChild(dropDown2);
    dropDowns.classList.add("dropdowns");

    let orderBy = document.createElement("div");
    let asc = document.createElement("img");
    asc.src = "../tree/images/ascending.svg";
    asc.addEventListener("click", updateAsc);

    function updateAsc(){
        model.set("sortOrder", "ASC");
        model.save_changes();
    }

    let dsc = document.createElement("img");
    dsc.src = "../tree/images/descending.svg";
    dsc.addEventListener("click", updateDsc);

    function updateDsc(){
        model.set("sortOrder", "DESC");
        model.save_changes();
    }

    orderBy.appendChild(asc);
    orderBy.appendChild(dsc);

    function addOption(parent, text, value, css = ""){
        let temp = document.createElement("option");
        temp.value = value;
        temp.innerHTML = text;
        if(css != ""){
            temp.classList.add(css);
        }
        parent.appendChild(temp);
    }

    el.appendChild(sort);
    el.appendChild(dropDowns);
    el.appendChild(orderBy);
}