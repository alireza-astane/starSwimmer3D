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
    sphere { m*<-0.9333922416363433,-1.1221653556263065,-0.6880525732863292>, 1 }        
    sphere {  m*<0.4953968153175896,-0.2751843834592256,9.173112281129015>, 1 }
    sphere {  m*<7.850748253317561,-0.3641046594535816,-5.4063810089163145>, 1 }
    sphere {  m*<-6.431332639179624,5.442152271639948,-3.5019971137313823>, 1}
    sphere { m*<-2.1670753700458736,-3.788433097697287,-1.2941790750134592>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4953968153175896,-0.2751843834592256,9.173112281129015>, <-0.9333922416363433,-1.1221653556263065,-0.6880525732863292>, 0.5 }
    cylinder { m*<7.850748253317561,-0.3641046594535816,-5.4063810089163145>, <-0.9333922416363433,-1.1221653556263065,-0.6880525732863292>, 0.5}
    cylinder { m*<-6.431332639179624,5.442152271639948,-3.5019971137313823>, <-0.9333922416363433,-1.1221653556263065,-0.6880525732863292>, 0.5 }
    cylinder {  m*<-2.1670753700458736,-3.788433097697287,-1.2941790750134592>, <-0.9333922416363433,-1.1221653556263065,-0.6880525732863292>, 0.5}

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
    sphere { m*<-0.9333922416363433,-1.1221653556263065,-0.6880525732863292>, 1 }        
    sphere {  m*<0.4953968153175896,-0.2751843834592256,9.173112281129015>, 1 }
    sphere {  m*<7.850748253317561,-0.3641046594535816,-5.4063810089163145>, 1 }
    sphere {  m*<-6.431332639179624,5.442152271639948,-3.5019971137313823>, 1}
    sphere { m*<-2.1670753700458736,-3.788433097697287,-1.2941790750134592>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4953968153175896,-0.2751843834592256,9.173112281129015>, <-0.9333922416363433,-1.1221653556263065,-0.6880525732863292>, 0.5 }
    cylinder { m*<7.850748253317561,-0.3641046594535816,-5.4063810089163145>, <-0.9333922416363433,-1.1221653556263065,-0.6880525732863292>, 0.5}
    cylinder { m*<-6.431332639179624,5.442152271639948,-3.5019971137313823>, <-0.9333922416363433,-1.1221653556263065,-0.6880525732863292>, 0.5 }
    cylinder {  m*<-2.1670753700458736,-3.788433097697287,-1.2941790750134592>, <-0.9333922416363433,-1.1221653556263065,-0.6880525732863292>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    