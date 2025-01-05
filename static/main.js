// main.js

// Wait for the DOM to load
document.addEventListener("DOMContentLoaded", function () {
    // Add event listeners for the form submission
    const form = document.querySelector("form");
    const loadingSpinner = document.getElementById("loadingSpinner");

    if (form) {
        form.addEventListener("submit", function (event) {
            const tickerInput = document.getElementById("ticker");
            const tickerValue = tickerInput.value.trim();

            // Validate ticker input
            if (!tickerValue || !/^[A-Za-z]+$/.test(tickerValue)) {
                event.preventDefault();
                alert("Please enter a valid stock ticker (e.g., AAPL, GOOG, TSLA).");
            } else {
                // Show loading spinner before the form is submitted
                loadingSpinner.style.display = "block";
            }
        });
    }

    // Hide the loading spinner once the page is loaded (after prediction is done)
    window.addEventListener("load", function () {
        loadingSpinner.style.display = "none";
    });

    // Smooth scroll to the graph section after prediction
    const graphContainer = document.querySelector(".graph-container");
    if (graphContainer) {
        const plotImage = graphContainer.querySelector("img");
        if (plotImage) {
            window.scrollTo({
                top: graphContainer.offsetTop - 20,
                behavior: "smooth",
            });
        }
    }

    // Highlight the navbar link (if added in the future) for the current section
    const navbarLinks = document.querySelectorAll(".navbar a");
    if (navbarLinks.length) {
        navbarLinks.forEach(link => {
            link.addEventListener("click", function (event) {
                navbarLinks.forEach(navLink => navLink.classList.remove("active"));
                this.classList.add("active");
            });
        });
    }

    // Add interactive hover effects on the stock summary table
    const tableRows = document.querySelectorAll("table tbody tr");
    tableRows.forEach(row => {
        row.addEventListener("mouseover", function () {
            this.style.backgroundColor = "#dff0ff"; // Light blue for hover
        });
        row.addEventListener("mouseout", function () {
            this.style.backgroundColor = ""; // Reset background
        });
    });
});
