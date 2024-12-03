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
    sphere { m*<-0.2383130058869063,-0.12263289152944223,-1.313197085225838>, 1 }        
    sphere {  m*<0.4411217047290604,0.2406298524548286,7.118678330466722>, 1 }
    sphere {  m*<2.496395388119351,-0.020598916143067983,-2.542406610677018>, 1 }
    sphere {  m*<-1.8599283657797963,2.2058410528891566,-2.2871428506418043>, 1}
    sphere { m*<-1.5921411447419644,-2.681850889514741,-2.0975965654792317>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4411217047290604,0.2406298524548286,7.118678330466722>, <-0.2383130058869063,-0.12263289152944223,-1.313197085225838>, 0.5 }
    cylinder { m*<2.496395388119351,-0.020598916143067983,-2.542406610677018>, <-0.2383130058869063,-0.12263289152944223,-1.313197085225838>, 0.5}
    cylinder { m*<-1.8599283657797963,2.2058410528891566,-2.2871428506418043>, <-0.2383130058869063,-0.12263289152944223,-1.313197085225838>, 0.5 }
    cylinder {  m*<-1.5921411447419644,-2.681850889514741,-2.0975965654792317>, <-0.2383130058869063,-0.12263289152944223,-1.313197085225838>, 0.5}

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
    sphere { m*<-0.2383130058869063,-0.12263289152944223,-1.313197085225838>, 1 }        
    sphere {  m*<0.4411217047290604,0.2406298524548286,7.118678330466722>, 1 }
    sphere {  m*<2.496395388119351,-0.020598916143067983,-2.542406610677018>, 1 }
    sphere {  m*<-1.8599283657797963,2.2058410528891566,-2.2871428506418043>, 1}
    sphere { m*<-1.5921411447419644,-2.681850889514741,-2.0975965654792317>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4411217047290604,0.2406298524548286,7.118678330466722>, <-0.2383130058869063,-0.12263289152944223,-1.313197085225838>, 0.5 }
    cylinder { m*<2.496395388119351,-0.020598916143067983,-2.542406610677018>, <-0.2383130058869063,-0.12263289152944223,-1.313197085225838>, 0.5}
    cylinder { m*<-1.8599283657797963,2.2058410528891566,-2.2871428506418043>, <-0.2383130058869063,-0.12263289152944223,-1.313197085225838>, 0.5 }
    cylinder {  m*<-1.5921411447419644,-2.681850889514741,-2.0975965654792317>, <-0.2383130058869063,-0.12263289152944223,-1.313197085225838>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    