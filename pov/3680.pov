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
    sphere { m*<0.02978511582957466,0.264182893692708,-0.11079314900038278>, 1 }        
    sphere {  m*<0.27052022057126635,0.3928929718730334,2.8767616221201684>, 1 }
    sphere {  m*<2.764493509835836,0.3662168690790826,-1.340002674451569>, 1 }
    sphere {  m*<-1.5918302440633165,2.5926568381113104,-1.0847389144163544>, 1}
    sphere { m*<-2.3317424343799233,-4.19994636745963,-1.4790473090557161>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.27052022057126635,0.3928929718730334,2.8767616221201684>, <0.02978511582957466,0.264182893692708,-0.11079314900038278>, 0.5 }
    cylinder { m*<2.764493509835836,0.3662168690790826,-1.340002674451569>, <0.02978511582957466,0.264182893692708,-0.11079314900038278>, 0.5}
    cylinder { m*<-1.5918302440633165,2.5926568381113104,-1.0847389144163544>, <0.02978511582957466,0.264182893692708,-0.11079314900038278>, 0.5 }
    cylinder {  m*<-2.3317424343799233,-4.19994636745963,-1.4790473090557161>, <0.02978511582957466,0.264182893692708,-0.11079314900038278>, 0.5}

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
    sphere { m*<0.02978511582957466,0.264182893692708,-0.11079314900038278>, 1 }        
    sphere {  m*<0.27052022057126635,0.3928929718730334,2.8767616221201684>, 1 }
    sphere {  m*<2.764493509835836,0.3662168690790826,-1.340002674451569>, 1 }
    sphere {  m*<-1.5918302440633165,2.5926568381113104,-1.0847389144163544>, 1}
    sphere { m*<-2.3317424343799233,-4.19994636745963,-1.4790473090557161>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.27052022057126635,0.3928929718730334,2.8767616221201684>, <0.02978511582957466,0.264182893692708,-0.11079314900038278>, 0.5 }
    cylinder { m*<2.764493509835836,0.3662168690790826,-1.340002674451569>, <0.02978511582957466,0.264182893692708,-0.11079314900038278>, 0.5}
    cylinder { m*<-1.5918302440633165,2.5926568381113104,-1.0847389144163544>, <0.02978511582957466,0.264182893692708,-0.11079314900038278>, 0.5 }
    cylinder {  m*<-2.3317424343799233,-4.19994636745963,-1.4790473090557161>, <0.02978511582957466,0.264182893692708,-0.11079314900038278>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    