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
    sphere { m*<-0.1406317936233149,-0.05796584093302104,-0.20953163091307653>, 1 }        
    sphere {  m*<0.1001033111183765,0.07074423724730416,2.778023140207473>, 1 }
    sphere {  m*<2.5940766003829467,0.044068134453353336,-1.4387411563642618>, 1 }
    sphere {  m*<-1.7622471535162076,2.2705081034855814,-1.1834773963290472>, 1}
    sphere { m*<-1.5544796466342707,-2.730642503495651,-1.0287061547910303>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.1001033111183765,0.07074423724730416,2.778023140207473>, <-0.1406317936233149,-0.05796584093302104,-0.20953163091307653>, 0.5 }
    cylinder { m*<2.5940766003829467,0.044068134453353336,-1.4387411563642618>, <-0.1406317936233149,-0.05796584093302104,-0.20953163091307653>, 0.5}
    cylinder { m*<-1.7622471535162076,2.2705081034855814,-1.1834773963290472>, <-0.1406317936233149,-0.05796584093302104,-0.20953163091307653>, 0.5 }
    cylinder {  m*<-1.5544796466342707,-2.730642503495651,-1.0287061547910303>, <-0.1406317936233149,-0.05796584093302104,-0.20953163091307653>, 0.5}

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
    sphere { m*<-0.1406317936233149,-0.05796584093302104,-0.20953163091307653>, 1 }        
    sphere {  m*<0.1001033111183765,0.07074423724730416,2.778023140207473>, 1 }
    sphere {  m*<2.5940766003829467,0.044068134453353336,-1.4387411563642618>, 1 }
    sphere {  m*<-1.7622471535162076,2.2705081034855814,-1.1834773963290472>, 1}
    sphere { m*<-1.5544796466342707,-2.730642503495651,-1.0287061547910303>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.1001033111183765,0.07074423724730416,2.778023140207473>, <-0.1406317936233149,-0.05796584093302104,-0.20953163091307653>, 0.5 }
    cylinder { m*<2.5940766003829467,0.044068134453353336,-1.4387411563642618>, <-0.1406317936233149,-0.05796584093302104,-0.20953163091307653>, 0.5}
    cylinder { m*<-1.7622471535162076,2.2705081034855814,-1.1834773963290472>, <-0.1406317936233149,-0.05796584093302104,-0.20953163091307653>, 0.5 }
    cylinder {  m*<-1.5544796466342707,-2.730642503495651,-1.0287061547910303>, <-0.1406317936233149,-0.05796584093302104,-0.20953163091307653>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    