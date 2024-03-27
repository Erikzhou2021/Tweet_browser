export function render({ model, el }) { 
    el.classList.add("sample-selector");

    let filePath = model.get("filePath");   
    let text1 = document.createElement("div");
    text1.innerHTML = "Displaying&nbsp;";
    let sampleSize = document.createElement("select");

    let option1 = document.createElement("option");
    option1.innerHTML = "50";
    option1.value = 50;
    sampleSize.appendChild(option1);

    let option2 = document.createElement("option");
    option2.innerHTML = "100";
    option2.value = 100;
    sampleSize.appendChild(option2);

    let option3 = document.createElement("option");
    option3.innerHTML = "150";
    option3.value = 150;
    sampleSize.appendChild(option3);

    let option4 = document.createElement("option");
    option4.innerHTML = "200";
    option4.value = 200;
    sampleSize.appendChild(option4);

    let option5 = document.createElement("option");
    option5.innerHTML = "All";
    option5.value = model.get("value");
    sampleSize.appendChild(option5);

    let text2 = document.createElement("div");
    text2.innerHTML = "posts from&nbsp;" + model.get("total").toString() + "&nbsp;results";

    function updateValue(){
        let val = sampleSize.value;
        model.set("value", val);
        model.save_changes();
    }

    sampleSize.value = model.get("value");
    sampleSize.addEventListener("change", updateValue);
    el.appendChild(text1);
    el.appendChild(sampleSize);
    el.appendChild(text2);
}