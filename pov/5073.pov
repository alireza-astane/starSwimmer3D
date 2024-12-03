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
    sphere { m*<-0.35093361848771115,-0.14150371279508758,-1.6418628545990892>, 1 }        
    sphere {  m*<0.5005380862566671,0.2900492687608021,8.312469193179165>, 1 }
    sphere {  m*<2.848883693767123,-0.02284825997815744,-3.099060747633574>, 1 }
    sphere {  m*<-1.9768751798768573,2.1870180773393435,-2.6084533194213786>, 1}
    sphere { m*<-1.7090879588390255,-2.700673865064554,-2.418907034258808>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5005380862566671,0.2900492687608021,8.312469193179165>, <-0.35093361848771115,-0.14150371279508758,-1.6418628545990892>, 0.5 }
    cylinder { m*<2.848883693767123,-0.02284825997815744,-3.099060747633574>, <-0.35093361848771115,-0.14150371279508758,-1.6418628545990892>, 0.5}
    cylinder { m*<-1.9768751798768573,2.1870180773393435,-2.6084533194213786>, <-0.35093361848771115,-0.14150371279508758,-1.6418628545990892>, 0.5 }
    cylinder {  m*<-1.7090879588390255,-2.700673865064554,-2.418907034258808>, <-0.35093361848771115,-0.14150371279508758,-1.6418628545990892>, 0.5}

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
    sphere { m*<-0.35093361848771115,-0.14150371279508758,-1.6418628545990892>, 1 }        
    sphere {  m*<0.5005380862566671,0.2900492687608021,8.312469193179165>, 1 }
    sphere {  m*<2.848883693767123,-0.02284825997815744,-3.099060747633574>, 1 }
    sphere {  m*<-1.9768751798768573,2.1870180773393435,-2.6084533194213786>, 1}
    sphere { m*<-1.7090879588390255,-2.700673865064554,-2.418907034258808>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5005380862566671,0.2900492687608021,8.312469193179165>, <-0.35093361848771115,-0.14150371279508758,-1.6418628545990892>, 0.5 }
    cylinder { m*<2.848883693767123,-0.02284825997815744,-3.099060747633574>, <-0.35093361848771115,-0.14150371279508758,-1.6418628545990892>, 0.5}
    cylinder { m*<-1.9768751798768573,2.1870180773393435,-2.6084533194213786>, <-0.35093361848771115,-0.14150371279508758,-1.6418628545990892>, 0.5 }
    cylinder {  m*<-1.7090879588390255,-2.700673865064554,-2.418907034258808>, <-0.35093361848771115,-0.14150371279508758,-1.6418628545990892>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    