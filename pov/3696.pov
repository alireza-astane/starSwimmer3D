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
    sphere { m*<0.019735369649850676,0.24518528937338524,-0.11661590851749609>, 1 }        
    sphere {  m*<0.2604704743915424,0.37389536755371067,2.8709388626030554>, 1 }
    sphere {  m*<2.754443763656112,0.34721926475975984,-1.345825433968682>, 1 }
    sphere {  m*<-1.6018799902430403,2.5736592337919877,-1.0905616739334674>, 1}
    sphere { m*<-2.2912492633757258,-4.123399833187503,-1.4555858213100366>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2604704743915424,0.37389536755371067,2.8709388626030554>, <0.019735369649850676,0.24518528937338524,-0.11661590851749609>, 0.5 }
    cylinder { m*<2.754443763656112,0.34721926475975984,-1.345825433968682>, <0.019735369649850676,0.24518528937338524,-0.11661590851749609>, 0.5}
    cylinder { m*<-1.6018799902430403,2.5736592337919877,-1.0905616739334674>, <0.019735369649850676,0.24518528937338524,-0.11661590851749609>, 0.5 }
    cylinder {  m*<-2.2912492633757258,-4.123399833187503,-1.4555858213100366>, <0.019735369649850676,0.24518528937338524,-0.11661590851749609>, 0.5}

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
    sphere { m*<0.019735369649850676,0.24518528937338524,-0.11661590851749609>, 1 }        
    sphere {  m*<0.2604704743915424,0.37389536755371067,2.8709388626030554>, 1 }
    sphere {  m*<2.754443763656112,0.34721926475975984,-1.345825433968682>, 1 }
    sphere {  m*<-1.6018799902430403,2.5736592337919877,-1.0905616739334674>, 1}
    sphere { m*<-2.2912492633757258,-4.123399833187503,-1.4555858213100366>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2604704743915424,0.37389536755371067,2.8709388626030554>, <0.019735369649850676,0.24518528937338524,-0.11661590851749609>, 0.5 }
    cylinder { m*<2.754443763656112,0.34721926475975984,-1.345825433968682>, <0.019735369649850676,0.24518528937338524,-0.11661590851749609>, 0.5}
    cylinder { m*<-1.6018799902430403,2.5736592337919877,-1.0905616739334674>, <0.019735369649850676,0.24518528937338524,-0.11661590851749609>, 0.5 }
    cylinder {  m*<-2.2912492633757258,-4.123399833187503,-1.4555858213100366>, <0.019735369649850676,0.24518528937338524,-0.11661590851749609>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    