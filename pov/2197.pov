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
    sphere { m*<1.1264545641004857,0.2641721401094155,0.5319022818839229>, 1 }        
    sphere {  m*<1.3706001070271792,0.2844424461522281,3.521881687788616>, 1 }
    sphere {  m*<3.863847296089716,0.2844424461522281,-0.6954005207020006>, 1 }
    sphere {  m*<-3.225135120547989,7.238739424252476,-2.041042698436218>, 1}
    sphere { m*<-3.7603039938179514,-7.967795759874334,-2.3567889181499773>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3706001070271792,0.2844424461522281,3.521881687788616>, <1.1264545641004857,0.2641721401094155,0.5319022818839229>, 0.5 }
    cylinder { m*<3.863847296089716,0.2844424461522281,-0.6954005207020006>, <1.1264545641004857,0.2641721401094155,0.5319022818839229>, 0.5}
    cylinder { m*<-3.225135120547989,7.238739424252476,-2.041042698436218>, <1.1264545641004857,0.2641721401094155,0.5319022818839229>, 0.5 }
    cylinder {  m*<-3.7603039938179514,-7.967795759874334,-2.3567889181499773>, <1.1264545641004857,0.2641721401094155,0.5319022818839229>, 0.5}

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
    sphere { m*<1.1264545641004857,0.2641721401094155,0.5319022818839229>, 1 }        
    sphere {  m*<1.3706001070271792,0.2844424461522281,3.521881687788616>, 1 }
    sphere {  m*<3.863847296089716,0.2844424461522281,-0.6954005207020006>, 1 }
    sphere {  m*<-3.225135120547989,7.238739424252476,-2.041042698436218>, 1}
    sphere { m*<-3.7603039938179514,-7.967795759874334,-2.3567889181499773>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3706001070271792,0.2844424461522281,3.521881687788616>, <1.1264545641004857,0.2641721401094155,0.5319022818839229>, 0.5 }
    cylinder { m*<3.863847296089716,0.2844424461522281,-0.6954005207020006>, <1.1264545641004857,0.2641721401094155,0.5319022818839229>, 0.5}
    cylinder { m*<-3.225135120547989,7.238739424252476,-2.041042698436218>, <1.1264545641004857,0.2641721401094155,0.5319022818839229>, 0.5 }
    cylinder {  m*<-3.7603039938179514,-7.967795759874334,-2.3567889181499773>, <1.1264545641004857,0.2641721401094155,0.5319022818839229>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    