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
    sphere { m*<0.804418055342045,0.7737203339461716,0.34149347458274826>, 1 }        
    sphere {  m*<1.0475117509182263,0.8418569178762197,3.3308490026159205>, 1 }
    sphere {  m*<3.540758939980762,0.8418569178762195,-0.8864332058746935>, 1 }
    sphere {  m*<-2.2033039487562993,5.323914142624051,-1.4368574089924728>, 1}
    sphere { m*<-3.884610269469191,-7.612048378620541,-2.430293566000028>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0475117509182263,0.8418569178762197,3.3308490026159205>, <0.804418055342045,0.7737203339461716,0.34149347458274826>, 0.5 }
    cylinder { m*<3.540758939980762,0.8418569178762195,-0.8864332058746935>, <0.804418055342045,0.7737203339461716,0.34149347458274826>, 0.5}
    cylinder { m*<-2.2033039487562993,5.323914142624051,-1.4368574089924728>, <0.804418055342045,0.7737203339461716,0.34149347458274826>, 0.5 }
    cylinder {  m*<-3.884610269469191,-7.612048378620541,-2.430293566000028>, <0.804418055342045,0.7737203339461716,0.34149347458274826>, 0.5}

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
    sphere { m*<0.804418055342045,0.7737203339461716,0.34149347458274826>, 1 }        
    sphere {  m*<1.0475117509182263,0.8418569178762197,3.3308490026159205>, 1 }
    sphere {  m*<3.540758939980762,0.8418569178762195,-0.8864332058746935>, 1 }
    sphere {  m*<-2.2033039487562993,5.323914142624051,-1.4368574089924728>, 1}
    sphere { m*<-3.884610269469191,-7.612048378620541,-2.430293566000028>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0475117509182263,0.8418569178762197,3.3308490026159205>, <0.804418055342045,0.7737203339461716,0.34149347458274826>, 0.5 }
    cylinder { m*<3.540758939980762,0.8418569178762195,-0.8864332058746935>, <0.804418055342045,0.7737203339461716,0.34149347458274826>, 0.5}
    cylinder { m*<-2.2033039487562993,5.323914142624051,-1.4368574089924728>, <0.804418055342045,0.7737203339461716,0.34149347458274826>, 0.5 }
    cylinder {  m*<-3.884610269469191,-7.612048378620541,-2.430293566000028>, <0.804418055342045,0.7737203339461716,0.34149347458274826>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    