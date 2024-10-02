export function render({ model, el }) {   
    let input = document.createElement('input');
    let placeholder = model.get("placeholder");
    if(placeholder != null && placeholder != ""){
        input.placeholder = placeholder;
    }
    input.value = '';
    input.classList.add("semantic-search");

    let fullHeader = document.createElement("div");
    fullHeader.classList.add("semantic-search-title");
    let header = document.createElement("h4");
    header.innerHTML = "Semantic Match";
    fullHeader.appendChild(header);
    let header2 = document.createElement("h5");
    header2.innerHTML = "(posts are about)";
    fullHeader.appendChild(header2);
    let slider = document.createElement("input");
    slider.type = "range";
    slider.min = "1";
    slider.max = "100";
    slider.classList.add("slider");
    fullHeader.appendChild(slider);
    let firstText = document.createElement("div");
    firstText.innerHTML = "Top ";
    let intBox = document.createElement("input");
    intBox.classList.add("int-box");
    intBox.min = "1";
    intBox.max = "100";
    intBox.value = 50;
    let secondText = document.createElement("div");
    secondText.innerHTML = "% Most Relavent";
    fullHeader.appendChild(firstText);
    fullHeader.appendChild(intBox);
    fullHeader.appendChild(secondText);

    input.addEventListener("change", (event) => {
        model.set("value", input.value);
        model.save_changes();
      });

    el.appendChild(fullHeader);
    el.appendChild(input);
}