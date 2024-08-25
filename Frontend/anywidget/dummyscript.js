export function render({ model, el }) {      
    let button = document.querySelector('.search-button');
    function preSearch(){
        let searchBars = document.querySelectorAll('.plusButton');
        searchBars.forEach((elem) =>{
            elem.click();
        });
        let invisButton = document.querySelector('.hidden-button');
        invisButton.click();
    } 
    if(button != null){
        button.addEventListener("click", preSearch);
    }
}