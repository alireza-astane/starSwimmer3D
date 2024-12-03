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
    sphere { m*<-0.6052944949536497,-0.8387534538436587,-0.5298683501940005>, 1 }        
    sphere {  m*<0.8138729992465122,0.15118546003625877,9.319421746841149>, 1 }
    sphere {  m*<8.18166019756931,-0.13390679075600342,-5.25125568223278>, 1 }
    sphere {  m*<-6.7143029961196765,6.389174582864637,-3.760448779051175>, 1}
    sphere { m*<-3.079131738293032,-6.226287933781163,-1.6754715767548456>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8138729992465122,0.15118546003625877,9.319421746841149>, <-0.6052944949536497,-0.8387534538436587,-0.5298683501940005>, 0.5 }
    cylinder { m*<8.18166019756931,-0.13390679075600342,-5.25125568223278>, <-0.6052944949536497,-0.8387534538436587,-0.5298683501940005>, 0.5}
    cylinder { m*<-6.7143029961196765,6.389174582864637,-3.760448779051175>, <-0.6052944949536497,-0.8387534538436587,-0.5298683501940005>, 0.5 }
    cylinder {  m*<-3.079131738293032,-6.226287933781163,-1.6754715767548456>, <-0.6052944949536497,-0.8387534538436587,-0.5298683501940005>, 0.5}

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
    sphere { m*<-0.6052944949536497,-0.8387534538436587,-0.5298683501940005>, 1 }        
    sphere {  m*<0.8138729992465122,0.15118546003625877,9.319421746841149>, 1 }
    sphere {  m*<8.18166019756931,-0.13390679075600342,-5.25125568223278>, 1 }
    sphere {  m*<-6.7143029961196765,6.389174582864637,-3.760448779051175>, 1}
    sphere { m*<-3.079131738293032,-6.226287933781163,-1.6754715767548456>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8138729992465122,0.15118546003625877,9.319421746841149>, <-0.6052944949536497,-0.8387534538436587,-0.5298683501940005>, 0.5 }
    cylinder { m*<8.18166019756931,-0.13390679075600342,-5.25125568223278>, <-0.6052944949536497,-0.8387534538436587,-0.5298683501940005>, 0.5}
    cylinder { m*<-6.7143029961196765,6.389174582864637,-3.760448779051175>, <-0.6052944949536497,-0.8387534538436587,-0.5298683501940005>, 0.5 }
    cylinder {  m*<-3.079131738293032,-6.226287933781163,-1.6754715767548456>, <-0.6052944949536497,-0.8387534538436587,-0.5298683501940005>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    