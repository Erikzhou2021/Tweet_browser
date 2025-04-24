export function render({ model, el }) { 
    el.classList.add("weight-by");

    let text1 = document.createElement("div");
    text1.classList.add("body2");
    text1.classList.add("medium");
    let tooltipText = "Select what attribute to weigh by for random sampling.\nIf 'likes' is selected, a post with 10 likes will be around 10 times as likely to be sampled as a post with 1 like.";
    let tooltip = '&nbsp; <img src="' + model.get("filePath") + 'info.svg"';
    tooltip += '" style="cursor:pointer;position:absolute;top:0px;" title="' + tooltipText + '">';
    text1.innerHTML = "Weighted By" + tooltip;
    let weightBy = document.createElement("select");

    let option1 = document.createElement("option");
    option1.innerHTML = "None";
    option1.value = "None";
    weightBy.appendChild(option1);

    let option2 = document.createElement("option");
    option2.innerHTML = "Influencer Score";
    option2.value = "SenderInfluencerScore";
    weightBy.appendChild(option2);

    let option3 = document.createElement("option");
    option3.innerHTML = "Retweets";
    option3.value = "Retweets";
    weightBy.appendChild(option3);

    let option4 = document.createElement("option");
    option4.innerHTML = "Star Rating";
    option4.value = "Star Rating";
    weightBy.appendChild(option4);

    let option5 = document.createElement("option");
    option5.innerHTML = "Followers";
    option5.value = "Sender Followers Count";
    weightBy.appendChild(option5);

    function updateModel(){
        model.set("value", weightBy.value);
        model.save_changes();
    }
    function updateValue(){
        weightBy.value = model.get("value");
    }

    weightBy.value = model.get("value");
    weightBy.addEventListener("change", updateModel);
    model.on("change:value", updateValue)
    el.appendChild(text1);
    el.appendChild(weightBy);
}