function post(data,url){
    console.log(data);
    console.log(url);
    $.post(url, data, function(response){
        $('.loader').addClass('hide');
        console.log('RESPONSE: ');
        res = response;
        console.log(res);

        $('.totComments .val').html((res['predDem']+res['predRep']).toString());
        $('.trueDem .val').html(res['countDem'].toString());
        $('.trueRep .val').html(res['countRep'].toString());
        if('valAcc' in res){
            $('.valAcc .val').html(res['valAcc']);
        }
        else{
            $('.valAcc .val').html('-');
        }
        $('.clfDem .val').html(res['predDem'].toString());
        $('.clfRep .val').html(res['predRep'].toString());

        var demFactor = (res['predDem']/(res['predDem']+res['predRep']))/0.5
        var repFactor = (res['predRep']/(res['predDem']+res['predRep']))/0.5
        
        $('.clfDem').width((410*demFactor).toString()+'px');
        $('.clfRep').width((410*repFactor).toString()+'px');

        var i = 0;
        demCommentHTML = ''
        for(i=0; i<res['demPredComments'].length; i++){
            demCommentHTML += '<div class="comment">'+res['demPredComments'][i].trim()+'</div>';
        }
        $('.commentContainer.dem').html(demCommentHTML);

        var i = 0;
        repCommentHTML = ''
        for(i=0; i<res['repPredComments'].length; i++){
            repCommentHTML += '<div class="comment">'+res['repPredComments'][i].trim()+'</div>';
        }
        $('.commentContainer.rep').html(repCommentHTML);
    });
}

function clfUsername(){
    $('.loader').removeClass('hide');   
    post({'username':$('#inp').val().trim()},'/username');
}

function clfComment(){
    $('.loader').removeClass('hide');
    post({'comment':$('#inp').val().trim()},'/comment');
}

$(document).ready(function(){
});