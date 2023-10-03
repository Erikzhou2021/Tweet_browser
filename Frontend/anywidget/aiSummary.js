export function render({ model, el }) {      
    el.classList.add("ai-summary"); 
    let header = document.createElement("h1");
    header.innerHTML = "AI Summary of Displayed Tweets";
    let tip = document.createElement("h3");
    tip.innerHTML = "Click each sentence to view the contributing tweets";
    let summary = document.createElement("div");
    summary.classList.add("summary");
    summary.innerHTML = "Blah blah blah blah";
    let caveat = document.createElement("h3");
    caveat.innerHTML = "*Not all tweets are captured in the summary. Click to view ommited tweets"
    el.appendChild(header);
    el.appendChild(tip);
    el.appendChild(summary);
    el.appendChild(caveat);
}