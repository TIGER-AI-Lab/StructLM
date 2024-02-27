window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true,
        autoplay: true,
        autoplaySpeed: 5000,
    }

    // Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    bulmaSlider.attach();

})


$(document).ready(function() {
    var trs = $('#tabResults1').children("[class!=th]")
    $('.buttonGroup').on('click', (e) => {
        // console.log(e.target.tagName)
        if (e.target.tagName !== 'BUTTON') {
            return
        }
        var tableId = $(e.currentTarget).attr('data-table-id');
        if (e.target.value === 'ALL') {
            $(tableId).children().show()
        } else {
            $(tableId).children().hide()
            $('#' + e.target.value).parent().nextUntil('.th').show()
            $('#' + e.target.value).parent().show()
        }
    })
    $('#myTable').DataTable({
        "pageLength": 50,
        "lengthChange": false
    });
});

$(document).ready(function() {
    var trs = $('#tabResults2').children("[class!=th]")
    $('.buttonGroup').on('click', (e) => {
        // console.log(e.target.tagName)
        if (e.target.tagName !== 'BUTTON') {
            return
        }
        var tableId = $(e.currentTarget).attr('data-table-id');
        if (e.target.value === 'ALL') {
            $(tableId).children().show()
        } else {
            $(tableId).children().hide()
            $('#' + e.target.value).parent().nextUntil('.th').show()
            $('#' + e.target.value).parent().show()
        }
    })
    $('#myTable').DataTable({
        "pageLength": 50,
        "lengthChange": false
    });
});