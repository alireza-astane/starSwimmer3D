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
    sphere { m*<-0.43994887582236397,-0.4786629804000879,-0.4532988526312555>, 1 }        
    sphere {  m*<0.9792186183777977,0.5112759334798294,9.395991244403891>, 1 }
    sphere {  m*<8.347005816700593,0.22618368268756694,-5.17468618467004>, 1 }
    sphere {  m*<-6.548957376988398,6.749265056308206,-3.6838792814884336>, 1}
    sphere { m*<-3.8544834213656545,-7.914852506037573,-2.0345272875696603>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9792186183777977,0.5112759334798294,9.395991244403891>, <-0.43994887582236397,-0.4786629804000879,-0.4532988526312555>, 0.5 }
    cylinder { m*<8.347005816700593,0.22618368268756694,-5.17468618467004>, <-0.43994887582236397,-0.4786629804000879,-0.4532988526312555>, 0.5}
    cylinder { m*<-6.548957376988398,6.749265056308206,-3.6838792814884336>, <-0.43994887582236397,-0.4786629804000879,-0.4532988526312555>, 0.5 }
    cylinder {  m*<-3.8544834213656545,-7.914852506037573,-2.0345272875696603>, <-0.43994887582236397,-0.4786629804000879,-0.4532988526312555>, 0.5}

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
    sphere { m*<-0.43994887582236397,-0.4786629804000879,-0.4532988526312555>, 1 }        
    sphere {  m*<0.9792186183777977,0.5112759334798294,9.395991244403891>, 1 }
    sphere {  m*<8.347005816700593,0.22618368268756694,-5.17468618467004>, 1 }
    sphere {  m*<-6.548957376988398,6.749265056308206,-3.6838792814884336>, 1}
    sphere { m*<-3.8544834213656545,-7.914852506037573,-2.0345272875696603>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9792186183777977,0.5112759334798294,9.395991244403891>, <-0.43994887582236397,-0.4786629804000879,-0.4532988526312555>, 0.5 }
    cylinder { m*<8.347005816700593,0.22618368268756694,-5.17468618467004>, <-0.43994887582236397,-0.4786629804000879,-0.4532988526312555>, 0.5}
    cylinder { m*<-6.548957376988398,6.749265056308206,-3.6838792814884336>, <-0.43994887582236397,-0.4786629804000879,-0.4532988526312555>, 0.5 }
    cylinder {  m*<-3.8544834213656545,-7.914852506037573,-2.0345272875696603>, <-0.43994887582236397,-0.4786629804000879,-0.4532988526312555>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    