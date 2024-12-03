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
    sphere { m*<-0.926208956978723,-1.1307938201597723,-0.6843757220692834>, 1 }        
    sphere {  m*<0.5022143688384635,-0.28084360637532035,9.176586468159337>, 1 }
    sphere {  m*<7.857565806838436,-0.3697638823696765,-5.402906821885994>, 1 }
    sphere {  m*<-6.461377067613035,5.471084003105684,-3.517331980687011>, 1}
    sphere { m*<-2.158657151559725,-3.7977752066729225,-1.2898748118163281>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5022143688384635,-0.28084360637532035,9.176586468159337>, <-0.926208956978723,-1.1307938201597723,-0.6843757220692834>, 0.5 }
    cylinder { m*<7.857565806838436,-0.3697638823696765,-5.402906821885994>, <-0.926208956978723,-1.1307938201597723,-0.6843757220692834>, 0.5}
    cylinder { m*<-6.461377067613035,5.471084003105684,-3.517331980687011>, <-0.926208956978723,-1.1307938201597723,-0.6843757220692834>, 0.5 }
    cylinder {  m*<-2.158657151559725,-3.7977752066729225,-1.2898748118163281>, <-0.926208956978723,-1.1307938201597723,-0.6843757220692834>, 0.5}

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
    sphere { m*<-0.926208956978723,-1.1307938201597723,-0.6843757220692834>, 1 }        
    sphere {  m*<0.5022143688384635,-0.28084360637532035,9.176586468159337>, 1 }
    sphere {  m*<7.857565806838436,-0.3697638823696765,-5.402906821885994>, 1 }
    sphere {  m*<-6.461377067613035,5.471084003105684,-3.517331980687011>, 1}
    sphere { m*<-2.158657151559725,-3.7977752066729225,-1.2898748118163281>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5022143688384635,-0.28084360637532035,9.176586468159337>, <-0.926208956978723,-1.1307938201597723,-0.6843757220692834>, 0.5 }
    cylinder { m*<7.857565806838436,-0.3697638823696765,-5.402906821885994>, <-0.926208956978723,-1.1307938201597723,-0.6843757220692834>, 0.5}
    cylinder { m*<-6.461377067613035,5.471084003105684,-3.517331980687011>, <-0.926208956978723,-1.1307938201597723,-0.6843757220692834>, 0.5 }
    cylinder {  m*<-2.158657151559725,-3.7977752066729225,-1.2898748118163281>, <-0.926208956978723,-1.1307938201597723,-0.6843757220692834>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    