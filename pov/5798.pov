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
    sphere { m*<-1.3097016630258067,-0.17623383700527492,-1.1839631780947482>, 1 }        
    sphere {  m*<0.0386416081779406,0.2803213490870853,8.714158482670916>, 1 }
    sphere {  m*<6.319415634373528,0.08844467995662744,-5.129476525309638>, 1 }
    sphere {  m*<-2.977542249383746,2.152932812662317,-2.074625595476048>, 1}
    sphere { m*<-2.7097550283459144,-2.7347591297415805,-1.8850793103134778>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.0386416081779406,0.2803213490870853,8.714158482670916>, <-1.3097016630258067,-0.17623383700527492,-1.1839631780947482>, 0.5 }
    cylinder { m*<6.319415634373528,0.08844467995662744,-5.129476525309638>, <-1.3097016630258067,-0.17623383700527492,-1.1839631780947482>, 0.5}
    cylinder { m*<-2.977542249383746,2.152932812662317,-2.074625595476048>, <-1.3097016630258067,-0.17623383700527492,-1.1839631780947482>, 0.5 }
    cylinder {  m*<-2.7097550283459144,-2.7347591297415805,-1.8850793103134778>, <-1.3097016630258067,-0.17623383700527492,-1.1839631780947482>, 0.5}

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
    sphere { m*<-1.3097016630258067,-0.17623383700527492,-1.1839631780947482>, 1 }        
    sphere {  m*<0.0386416081779406,0.2803213490870853,8.714158482670916>, 1 }
    sphere {  m*<6.319415634373528,0.08844467995662744,-5.129476525309638>, 1 }
    sphere {  m*<-2.977542249383746,2.152932812662317,-2.074625595476048>, 1}
    sphere { m*<-2.7097550283459144,-2.7347591297415805,-1.8850793103134778>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.0386416081779406,0.2803213490870853,8.714158482670916>, <-1.3097016630258067,-0.17623383700527492,-1.1839631780947482>, 0.5 }
    cylinder { m*<6.319415634373528,0.08844467995662744,-5.129476525309638>, <-1.3097016630258067,-0.17623383700527492,-1.1839631780947482>, 0.5}
    cylinder { m*<-2.977542249383746,2.152932812662317,-2.074625595476048>, <-1.3097016630258067,-0.17623383700527492,-1.1839631780947482>, 0.5 }
    cylinder {  m*<-2.7097550283459144,-2.7347591297415805,-1.8850793103134778>, <-1.3097016630258067,-0.17623383700527492,-1.1839631780947482>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    