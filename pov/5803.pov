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
    sphere { m*<-1.316673566455564,-0.17646762911752756,-1.1801484879532564>, 1 }        
    sphere {  m*<0.03473771772563422,0.2802388992777467,8.71754754536131>, 1 }
    sphere {  m*<6.34229605552734,0.08913036331911159,-5.14410488351832>, 1 }
    sphere {  m*<-2.984761765599093,2.1527039122507987,-2.0703342452024383>, 1}
    sphere { m*<-2.7169745445612614,-2.7349880301530987,-1.8807879600398678>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.03473771772563422,0.2802388992777467,8.71754754536131>, <-1.316673566455564,-0.17646762911752756,-1.1801484879532564>, 0.5 }
    cylinder { m*<6.34229605552734,0.08913036331911159,-5.14410488351832>, <-1.316673566455564,-0.17646762911752756,-1.1801484879532564>, 0.5}
    cylinder { m*<-2.984761765599093,2.1527039122507987,-2.0703342452024383>, <-1.316673566455564,-0.17646762911752756,-1.1801484879532564>, 0.5 }
    cylinder {  m*<-2.7169745445612614,-2.7349880301530987,-1.8807879600398678>, <-1.316673566455564,-0.17646762911752756,-1.1801484879532564>, 0.5}

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
    sphere { m*<-1.316673566455564,-0.17646762911752756,-1.1801484879532564>, 1 }        
    sphere {  m*<0.03473771772563422,0.2802388992777467,8.71754754536131>, 1 }
    sphere {  m*<6.34229605552734,0.08913036331911159,-5.14410488351832>, 1 }
    sphere {  m*<-2.984761765599093,2.1527039122507987,-2.0703342452024383>, 1}
    sphere { m*<-2.7169745445612614,-2.7349880301530987,-1.8807879600398678>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.03473771772563422,0.2802388992777467,8.71754754536131>, <-1.316673566455564,-0.17646762911752756,-1.1801484879532564>, 0.5 }
    cylinder { m*<6.34229605552734,0.08913036331911159,-5.14410488351832>, <-1.316673566455564,-0.17646762911752756,-1.1801484879532564>, 0.5}
    cylinder { m*<-2.984761765599093,2.1527039122507987,-2.0703342452024383>, <-1.316673566455564,-0.17646762911752756,-1.1801484879532564>, 0.5 }
    cylinder {  m*<-2.7169745445612614,-2.7349880301530987,-1.8807879600398678>, <-1.316673566455564,-0.17646762911752756,-1.1801484879532564>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    