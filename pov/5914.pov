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
    sphere { m*<-1.4718595941120842,-0.18161512382508663,-1.0937790391500668>, 1 }        
    sphere {  m*<-0.05421138600542627,0.2783496540808074,8.794491107851334>, 1 }
    sphere {  m*<6.847881608694109,0.10416072148692895,-5.470476799077768>, 1 }
    sphere {  m*<-3.1452601729969367,2.147664770092521,-1.9736503278781972>, 1}
    sphere { m*<-2.8774729519591054,-2.7400271723113763,-1.7841040427156267>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.05421138600542627,0.2783496540808074,8.794491107851334>, <-1.4718595941120842,-0.18161512382508663,-1.0937790391500668>, 0.5 }
    cylinder { m*<6.847881608694109,0.10416072148692895,-5.470476799077768>, <-1.4718595941120842,-0.18161512382508663,-1.0937790391500668>, 0.5}
    cylinder { m*<-3.1452601729969367,2.147664770092521,-1.9736503278781972>, <-1.4718595941120842,-0.18161512382508663,-1.0937790391500668>, 0.5 }
    cylinder {  m*<-2.8774729519591054,-2.7400271723113763,-1.7841040427156267>, <-1.4718595941120842,-0.18161512382508663,-1.0937790391500668>, 0.5}

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
    sphere { m*<-1.4718595941120842,-0.18161512382508663,-1.0937790391500668>, 1 }        
    sphere {  m*<-0.05421138600542627,0.2783496540808074,8.794491107851334>, 1 }
    sphere {  m*<6.847881608694109,0.10416072148692895,-5.470476799077768>, 1 }
    sphere {  m*<-3.1452601729969367,2.147664770092521,-1.9736503278781972>, 1}
    sphere { m*<-2.8774729519591054,-2.7400271723113763,-1.7841040427156267>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.05421138600542627,0.2783496540808074,8.794491107851334>, <-1.4718595941120842,-0.18161512382508663,-1.0937790391500668>, 0.5 }
    cylinder { m*<6.847881608694109,0.10416072148692895,-5.470476799077768>, <-1.4718595941120842,-0.18161512382508663,-1.0937790391500668>, 0.5}
    cylinder { m*<-3.1452601729969367,2.147664770092521,-1.9736503278781972>, <-1.4718595941120842,-0.18161512382508663,-1.0937790391500668>, 0.5 }
    cylinder {  m*<-2.8774729519591054,-2.7400271723113763,-1.7841040427156267>, <-1.4718595941120842,-0.18161512382508663,-1.0937790391500668>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    