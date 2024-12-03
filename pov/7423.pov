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
    sphere { m*<-0.6101862910671555,-0.8494068306174519,-0.5321336800682266>, 1 }        
    sphere {  m*<0.8089812031330065,0.14053208326246547,9.317156416966922>, 1 }
    sphere {  m*<8.176768401455801,-0.14456016752979672,-5.253521012107006>, 1 }
    sphere {  m*<-6.719194792233181,6.378521206090845,-3.762714108925401>, 1}
    sphere { m*<-3.0553437386076383,-6.174482315259224,-1.6644556502760972>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8089812031330065,0.14053208326246547,9.317156416966922>, <-0.6101862910671555,-0.8494068306174519,-0.5321336800682266>, 0.5 }
    cylinder { m*<8.176768401455801,-0.14456016752979672,-5.253521012107006>, <-0.6101862910671555,-0.8494068306174519,-0.5321336800682266>, 0.5}
    cylinder { m*<-6.719194792233181,6.378521206090845,-3.762714108925401>, <-0.6101862910671555,-0.8494068306174519,-0.5321336800682266>, 0.5 }
    cylinder {  m*<-3.0553437386076383,-6.174482315259224,-1.6644556502760972>, <-0.6101862910671555,-0.8494068306174519,-0.5321336800682266>, 0.5}

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
    sphere { m*<-0.6101862910671555,-0.8494068306174519,-0.5321336800682266>, 1 }        
    sphere {  m*<0.8089812031330065,0.14053208326246547,9.317156416966922>, 1 }
    sphere {  m*<8.176768401455801,-0.14456016752979672,-5.253521012107006>, 1 }
    sphere {  m*<-6.719194792233181,6.378521206090845,-3.762714108925401>, 1}
    sphere { m*<-3.0553437386076383,-6.174482315259224,-1.6644556502760972>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8089812031330065,0.14053208326246547,9.317156416966922>, <-0.6101862910671555,-0.8494068306174519,-0.5321336800682266>, 0.5 }
    cylinder { m*<8.176768401455801,-0.14456016752979672,-5.253521012107006>, <-0.6101862910671555,-0.8494068306174519,-0.5321336800682266>, 0.5}
    cylinder { m*<-6.719194792233181,6.378521206090845,-3.762714108925401>, <-0.6101862910671555,-0.8494068306174519,-0.5321336800682266>, 0.5 }
    cylinder {  m*<-3.0553437386076383,-6.174482315259224,-1.6644556502760972>, <-0.6101862910671555,-0.8494068306174519,-0.5321336800682266>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    