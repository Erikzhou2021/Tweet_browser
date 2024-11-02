export function render({ model, el }) {   
    let input = document.createElement('input');
    let placeholder = model.get("placeholder");
    if(placeholder != null && placeholder != ""){
        input.placeholder = placeholder;
    }
    input.value = model.get("value");
    input.classList.add("semantic-search-box");

    let fullHeader = document.createElement("div");
    fullHeader.classList.add("semantic-search-title");
    let header = document.createElement("div");
    header.innerHTML = "Semantic Match";
    header.classList.add("heading5");
    header.classList.add("medium");
    fullHeader.appendChild(header);
    let header2 = document.createElement("div");
    header2.innerHTML = "&nbsp; (posts are about)";
    header2.classList.add("body0");
    fullHeader.appendChild(header2);
    let slider = document.createElement("input");
    slider.type = "range";
    slider.min = "1";
    slider.max = "100";
    slider.classList.add("int-slider");
    fullHeader.appendChild(slider);
    let firstText = document.createElement("div");
    firstText.innerHTML = "Top ";
    let intBox = document.createElement("input");
    intBox.classList.add("int-box");
    intBox.min = "1";
    intBox.max = "100";
    intBox.value = model.get("filterPercent");
    let secondText = document.createElement("div");
    secondText.innerHTML = "% Most Relavent";
    fullHeader.appendChild(firstText);
    fullHeader.appendChild(intBox);
    fullHeader.appendChild(secondText);

    slider.addEventListener("input", (event) =>{
        intBox.value = slider.value;
        model.set("filterPercent", intBox.value);
        model.save_changes();
    })

    intBox.addEventListener("input", (event) =>{
        slider.value = intBox.value;
        model.set("filterPercent", intBox.value);
        model.save_changes();
    })

    input.addEventListener("change", (event) => {
        model.set("value", input.value);
        model.save_changes();
      });

    el.classList.add("semantic-search");
    el.appendChild(fullHeader);
    el.appendChild(input);
}