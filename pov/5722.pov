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
    sphere { m*<-1.2039718249132774,-0.17266012456284835,-1.2410836789221757>, 1 }        
    sphere {  m*<0.09685689741855469,0.281546424926455,8.663505932878952>, 1 }
    sphere {  m*<5.970403126272217,0.07792258724965603,-4.907963895372844>, 1 }
    sphere {  m*<-2.8679601230571214,2.156432200787599,-2.139115829035085>, 1}
    sphere { m*<-2.60017290201929,-2.7312597416162983,-1.9495695438725145>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.09685689741855469,0.281546424926455,8.663505932878952>, <-1.2039718249132774,-0.17266012456284835,-1.2410836789221757>, 0.5 }
    cylinder { m*<5.970403126272217,0.07792258724965603,-4.907963895372844>, <-1.2039718249132774,-0.17266012456284835,-1.2410836789221757>, 0.5}
    cylinder { m*<-2.8679601230571214,2.156432200787599,-2.139115829035085>, <-1.2039718249132774,-0.17266012456284835,-1.2410836789221757>, 0.5 }
    cylinder {  m*<-2.60017290201929,-2.7312597416162983,-1.9495695438725145>, <-1.2039718249132774,-0.17266012456284835,-1.2410836789221757>, 0.5}

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
    sphere { m*<-1.2039718249132774,-0.17266012456284835,-1.2410836789221757>, 1 }        
    sphere {  m*<0.09685689741855469,0.281546424926455,8.663505932878952>, 1 }
    sphere {  m*<5.970403126272217,0.07792258724965603,-4.907963895372844>, 1 }
    sphere {  m*<-2.8679601230571214,2.156432200787599,-2.139115829035085>, 1}
    sphere { m*<-2.60017290201929,-2.7312597416162983,-1.9495695438725145>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.09685689741855469,0.281546424926455,8.663505932878952>, <-1.2039718249132774,-0.17266012456284835,-1.2410836789221757>, 0.5 }
    cylinder { m*<5.970403126272217,0.07792258724965603,-4.907963895372844>, <-1.2039718249132774,-0.17266012456284835,-1.2410836789221757>, 0.5}
    cylinder { m*<-2.8679601230571214,2.156432200787599,-2.139115829035085>, <-1.2039718249132774,-0.17266012456284835,-1.2410836789221757>, 0.5 }
    cylinder {  m*<-2.60017290201929,-2.7312597416162983,-1.9495695438725145>, <-1.2039718249132774,-0.17266012456284835,-1.2410836789221757>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    