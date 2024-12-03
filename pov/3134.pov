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
    sphere { m*<0.41027861622765993,0.9834513047915692,0.10966238400727175>, 1 }        
    sphere {  m*<0.6510137209693516,1.1121613829718948,3.0972171551278227>, 1 }
    sphere {  m*<3.144987010233916,1.0854852801779438,-1.1195471414439124>, 1 }
    sphere {  m*<-1.21133674366523,3.3119252492101703,-0.8642833814086982>, 1}
    sphere { m*<-3.676025982883192,-6.741121685794054,-2.257916713890889>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6510137209693516,1.1121613829718948,3.0972171551278227>, <0.41027861622765993,0.9834513047915692,0.10966238400727175>, 0.5 }
    cylinder { m*<3.144987010233916,1.0854852801779438,-1.1195471414439124>, <0.41027861622765993,0.9834513047915692,0.10966238400727175>, 0.5}
    cylinder { m*<-1.21133674366523,3.3119252492101703,-0.8642833814086982>, <0.41027861622765993,0.9834513047915692,0.10966238400727175>, 0.5 }
    cylinder {  m*<-3.676025982883192,-6.741121685794054,-2.257916713890889>, <0.41027861622765993,0.9834513047915692,0.10966238400727175>, 0.5}

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
    sphere { m*<0.41027861622765993,0.9834513047915692,0.10966238400727175>, 1 }        
    sphere {  m*<0.6510137209693516,1.1121613829718948,3.0972171551278227>, 1 }
    sphere {  m*<3.144987010233916,1.0854852801779438,-1.1195471414439124>, 1 }
    sphere {  m*<-1.21133674366523,3.3119252492101703,-0.8642833814086982>, 1}
    sphere { m*<-3.676025982883192,-6.741121685794054,-2.257916713890889>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6510137209693516,1.1121613829718948,3.0972171551278227>, <0.41027861622765993,0.9834513047915692,0.10966238400727175>, 0.5 }
    cylinder { m*<3.144987010233916,1.0854852801779438,-1.1195471414439124>, <0.41027861622765993,0.9834513047915692,0.10966238400727175>, 0.5}
    cylinder { m*<-1.21133674366523,3.3119252492101703,-0.8642833814086982>, <0.41027861622765993,0.9834513047915692,0.10966238400727175>, 0.5 }
    cylinder {  m*<-3.676025982883192,-6.741121685794054,-2.257916713890889>, <0.41027861622765993,0.9834513047915692,0.10966238400727175>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    