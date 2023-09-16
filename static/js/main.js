$(document).ready(function() {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function(e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function() {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function() {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function(data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                if (data == 'Potato___Early_blight') {

                    $('#result').text(' Result:  ' + data);
                    $('#remedies').text("Remedies: # Burn or bag infected plant parts. Do NOT compost. # Drip irrigation and soaker hoses can be used to help keep the foliage dry. ");
                } else if (data == 'Potato___Late_blight') {
                    $('#result').text(' Result:  ' + data);
                    $('#remedies').text("Remedies: # Monitor the field, remove and destroy infected leaves. # Treat organically with copper spray. # - Use chemical fungicides,the best of which for potato is chlorothalonil.");
                } else
                    $('#result').text(' Result:  ' + data);


                console.log('Success!');
            },
        });
    });

});