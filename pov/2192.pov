#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<1.1303403024030332,0.2576226439748341,0.534199798626152>, 1 }        
    sphere {  m*<1.374491174742626,0.2773568741269526,3.52418237798205>, 1 }
    sphere {  m*<3.867738363805163,0.2773568741269526,-0.6930998305085669>, 1 }
    sphere {  m*<-3.237063071981327,7.262100614633704,-2.048095464768892>, 1}
    sphere { m*<-3.7586112236203237,-7.972587443697703,-2.355787953407389>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.374491174742626,0.2773568741269526,3.52418237798205>, <1.1303403024030332,0.2576226439748341,0.534199798626152>, 0.5 }
    cylinder { m*<3.867738363805163,0.2773568741269526,-0.6930998305085669>, <1.1303403024030332,0.2576226439748341,0.534199798626152>, 0.5}
    cylinder { m*<-3.237063071981327,7.262100614633704,-2.048095464768892>, <1.1303403024030332,0.2576226439748341,0.534199798626152>, 0.5 }
    cylinder {  m*<-3.7586112236203237,-7.972587443697703,-2.355787953407389>, <1.1303403024030332,0.2576226439748341,0.534199798626152>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<1.1303403024030332,0.2576226439748341,0.534199798626152>, 1 }        
    sphere {  m*<1.374491174742626,0.2773568741269526,3.52418237798205>, 1 }
    sphere {  m*<3.867738363805163,0.2773568741269526,-0.6930998305085669>, 1 }
    sphere {  m*<-3.237063071981327,7.262100614633704,-2.048095464768892>, 1}
    sphere { m*<-3.7586112236203237,-7.972587443697703,-2.355787953407389>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.374491174742626,0.2773568741269526,3.52418237798205>, <1.1303403024030332,0.2576226439748341,0.534199798626152>, 0.5 }
    cylinder { m*<3.867738363805163,0.2773568741269526,-0.6930998305085669>, <1.1303403024030332,0.2576226439748341,0.534199798626152>, 0.5}
    cylinder { m*<-3.237063071981327,7.262100614633704,-2.048095464768892>, <1.1303403024030332,0.2576226439748341,0.534199798626152>, 0.5 }
    cylinder {  m*<-3.7586112236203237,-7.972587443697703,-2.355787953407389>, <1.1303403024030332,0.2576226439748341,0.534199798626152>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    